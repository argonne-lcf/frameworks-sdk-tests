import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import cond, fori_loop
import jax
from jax import custom_jvp
from functools import partial

import logging
logger = logging.getLogger()

class Asymm(object):
    def __init__(self, name, bf_size, util, config):
        self.util = util
        self.backflow_single_size = bf_size[0]
        self.backflow_paired_size = bf_size[1]

        self.npart = config.npart
        self.ndense = config.ndense
        self.nsingle = config.nsingle
        self.a = config.amplitude_limit
        self.nhid = config.nhid

        if name == 'pfaffian':
            self.dense_depth = config.pfaffian_phi_depth
            self.build = self.build_pfaffian
            self.run = self.logphasepsi_pfaffian

            # Compute pfaffian using Skew-symmetric Rho
            if config.pfaffian_compute == 'Parlett-Reid':
                self.pfaffian_compute = pfaffian_parlett_reid
            elif config.pfaffian_compute == 'Householder':
                self.pfaffian_compute = pfaffian_householder
            else:
                logger.error('Invalid Pfaffian algorithm. Please choose either "Parlett-Reid" or "Householder".')
                raise ValueError('Invalid Pfaffian algorithm. Please choose either "Parlett-Reid" or "Householder".')

        elif name == 'hidden':
            self.dense_depth = config.hidden_orb_depth
            self.build = self.build_hidden
            self.run = self.logphasepsi_hidden
        else:
            raise NameError(f'Undefined wavefunction type: {name}')

    def build_pfaffian(self):
        # pfaffian pairing orbital
        phi_init, self.phi_apply = self.util.dense_net(
            self.ndense, self.dense_depth, 1
            )

        phi_pf_in_size = self.backflow_paired_size
        phi_sp_in_size = self.backflow_single_size

        # Initialize backflow and Jastrow parameters
        # Jastrow output size is a single real number
        jastrow_params = self.util.deepset_build(self.backflow_paired_size, 1)

        # Initialize pfaffian pairing orbital parameters for the phase and amplitude
        phi_pf_p_params, phi_pf_a_params = self.util.init_params(
            phi_init, phi_pf_in_size, count=2
            )

        if self.npart % 2 == 1:
            # If there are an odd number of particles we need to add a single
            # particle orbital because the pfaffian is zero for odd size matricies
            rho_sp_p_params, rho_sp_a_params = self.util.init_params(
                phi_init, phi_sp_in_size, count=2
                )

        # Save parameters
        params = [jastrow_params, phi_pf_p_params, phi_pf_a_params]
        if self.npart % 2 == 1:
            params += [rho_sp_p_params, rho_sp_a_params]

        return params

    @partial(jit, static_argnums=(0,))
    def logphasepsi_pfaffian(self, params, x, x_pair):
        # Separate parameters
        jastrow_params, rho_pf_p_params, rho_pf_a_params = params[0:3]

        if self.npart % 2 == 1:
            rho_sp_p_params, rho_sp_a_params = params[3:5]

        # Symmetric Jastrow type function
        # When we only have one deepset we use the single_deepset function.
        list_x_pair = jnp.reshape(x_pair, shape=(-1, x_pair.shape[-1]))
        rho_j_p, rho_j_a = self.util.single_deepset(*jastrow_params, list_x_pair)

        # Antisymmetric
        # Create npart by npart matrix, Rho, to take the pfaffian of

        # Trainable pairing orbital
        # Creates npart x npart elements of Rho
        pf_p = self.phi_apply(rho_pf_p_params, x_pair)
        pf_a = self.phi_apply(rho_pf_a_params, x_pair)
        pf = jnp.exp(self.a*jnp.tanh(pf_a/self.a) + 1j * pf_p)

        # pf is list of npart*(npart-1) elements convert to matrix
        # with shape npart, npart with zeros on diagonal
        Rho_reshape = jnp.reshape(pf, (self.npart-1, self.npart))
        Rho_reshape = jnp.pad(Rho_reshape, ((0, 0), (1, 0)), constant_values=0.0+0.0j)
        Rho_reshape = jnp.reshape(Rho_reshape, (self.npart**2-1,))
        Rho_reshape = jnp.pad(Rho_reshape, (0, 1), constant_values=0.0+0.0j)
        Rho = jnp.reshape(Rho_reshape, (self.npart, self.npart))

        # If npart is odd the matrix is instead npart+1 by npart+1
        if self.npart % 2 == 1:
            # Odd number of particles
            sp_p = self.phi_apply(rho_sp_p_params, x)
            sp_a = self.phi_apply(rho_sp_a_params, x)
            pf_sp = jnp.exp(self.a*jnp.tanh(sp_a/self.a) + 1j * sp_p)
            pf_sp = jnp.reshape(pf_sp, (self.npart,))

            # Set the npart+1 column based on unpaired particles
            # Rho will be made skew-symmetric later
            Rho = jnp.pad(Rho, ((0, 1), (0, 1)), constant_values=0.0+0.0j)
            Rho = Rho.at[:self.npart, self.npart].set(pf_sp)

        pf_p, pf_a = self.pfaffian_compute(Rho - jnp.transpose(Rho))
     
        # sign and log are always real
        phase = pf_p + rho_j_p
        log = pf_a + rho_j_a

        phase = jnp.reshape(phase, ())
        log = jnp.reshape(log, ())
        return phase, log

    def build_hidden(self):
        # Feed-forward constructor for single-particle orbitals wave function
        orb_init, self.orb_apply = self.util.dense_net(
                self.nsingle, self.dense_depth, 1
                )

        x_i_size = self.backflow_single_size

        # Initialize parameters
        # Create nhid parallel structures of rho(phi(x))
        # rho is composed of two parallel networks with outputs npart and nhid
        deepset_params = self.util.deepset_build(x_i_size, (self.npart, self.nhid), num=self.nhid)

        orb_h_params_i = self.util.init_params(
            orb_init, x_i_size, count=2*self.nhid
            )
        orb_h_p_params = orb_h_params_i[0::2]
        orb_h_a_params = orb_h_params_i[1::2]

        orb_v_params_i = self.util.init_params(
            orb_init, x_i_size, count=2*self.npart
            )
        orb_v_p_params = orb_v_params_i[0::2]
        orb_v_a_params = orb_v_params_i[1::2]

        params = [deepset_params, orb_h_p_params, orb_h_a_params, orb_v_p_params, orb_v_a_params]

        return params

    @partial(jit, static_argnums=(0,))
    def logphasepsi_hidden(self, params, x, x_pair):
        deepset_params, orb_h_p_params, orb_h_a_params, orb_v_p_params, orb_v_a_params = params

        # rho_h is (nhid, npart + nhid )
        # deepset_run will run all of our nhid deepsets (which each have output
        # size of npart + nhid) together on x_i
        rho_p_h, rho_a_h = self.util.deepset_run(deepset_params, x)
        rho_h = jnp.exp(self.a * jnp.tanh(rho_a_h / self.a) + 1j * rho_p_h)

        # Slater
        # orb_v is (nstate=npart, npart)
        orb_v_p_params_stacked = self.util.pytrees_stack(orb_v_p_params)
        orb_v_a_params_stacked = self.util.pytrees_stack(orb_v_a_params)
        orb_v_p = jnp.reshape(vmap(self.orb_apply, in_axes=(0, None))(orb_v_p_params_stacked, x), shape=(self.npart, self.npart))
        orb_v_a = jnp.reshape(vmap(self.orb_apply, in_axes=(0, None))(orb_v_a_params_stacked, x), shape=(self.npart, self.npart))
        orb_v = jnp.exp(self.a*jnp.tanh(orb_v_a/self.a)+ 1j * orb_v_p)

        # orb_h is (nstate=nhid, npart)
        # We map over the stacked parameters
        # Each chi function returns a single result for each particle and there are nhid chi functions.
        orb_h_p_params_stacked = self.util.pytrees_stack(orb_h_p_params)
        orb_h_a_params_stacked = self.util.pytrees_stack(orb_h_a_params)
        orb_h_p = jnp.reshape(vmap(self.orb_apply, in_axes=(0, None))(orb_h_p_params_stacked, x), shape=(self.nhid, self.npart))
        orb_h_a = jnp.reshape(vmap(self.orb_apply, in_axes=(0, None))(orb_h_a_params_stacked, x), shape=(self.nhid, self.npart))
        orb_h = jnp.exp(self.a*jnp.tanh(orb_h_a/self.a)+ 1j * orb_h_p)

        phase, logdet = self.phi_spin(orb_v, orb_h, rho_h)

        log = logdet
        return phase, log

    @partial(jit, static_argnums=(0,))
    def phi_spin(self, Rnl_v, Rnl_h, rho_h):
        ph = jnp.zeros((self.npart + self.nhid, self.npart + self.nhid), dtype=jnp.complex128)
        # Visible orbitals, visible coordinates
        for i in range(self.npart):
            ph = ph.at[i, 0:self.npart].set(Rnl_v[i, :])
        for i in range(self.nhid):
            # Hidden orbitals, visible coordinates
            ph = ph.at[self.npart+i, 0:self.npart].set(Rnl_h[i, :])
            # Hidden orbitals, hidden coordinates
            ph = ph.at[:, self.npart+i].set(rho_h[i, :])
        sign, logdet = jnp.linalg.slogdet(ph)
        phase = jnp.imag(jnp.log(sign))
        return phase, logdet

def register_pfaffian_jvp(primal_fn):
    """
    A decorator that registers a custom JVP for a log-Pfaffian function.

    This is useful because the analytical derivative of the log-Pfaffian is
    known, stable, and efficient to compute. This rule is the same regardless
    of the numerical method used for the forward pass (the "primal").

    Args:
        primal_fn: A function with the signature `func(A) -> (im_logpf, re_logpf)`
                   that computes the phase and log-amplitude of the Pfaffian.

    Returns:
        A new function with the custom JVP rule attached.
    """
    @custom_jvp
    def pfaffian_with_jvp(A):
        return primal_fn(A)  # ← use the value fn in the primal

    @pfaffian_with_jvp.defjvp
    def pfaffian_jvp_rule(primals, tangents):
        (A,), (A_dot,) = primals, tangents

        # PRIMAL: call the raw value fn, NOT the wrapped one
        primal_out = primal_fn(A)

        # TANGENT: d log pf = 1/2 Tr(As^{-1} dAs) with tiny reg
        As     = 0.5 * (A - A.T)
        As_dot = 0.5 * (A_dot - A_dot.T)
        n = As.shape[0]
        eps = jnp.finfo(As.real.dtype).eps
        lam = jnp.sqrt(eps) * jnp.linalg.norm(As)
        I = jnp.eye(n, dtype=As.dtype)
        X = jnp.linalg.solve(As + lam * I, As_dot)
        z = 0.5 * jnp.trace(X)
        return primal_out, (jnp.imag(z), jnp.real(z))

    return pfaffian_with_jvp

@register_pfaffian_jvp
def pfaffian_parlett_reid(A):
    n = A.shape[0]
    pf_A = jnp.array(1.0 + 0.0j, dtype=A.dtype)

    def do_pivot(ops):
        A, k, kp = ops
        idx  = jnp.arange(n)
        perm = idx.at[k+1].set(kp).at[kp].set(k+1)
        A_sw = A[perm, :][:, perm]
        return A_sw, -1.0

    def no_pivot(ops):
        A, k, kp = ops
        return A, 1.0

    for k in range(0, n-1, 2):
        # pick kp maximizing |A[j,k]| for j>=k+1
        tail = A[k+1:, k]
        kp = k + 1 + jnp.argmax(jnp.abs(tail))

        # symmetric swap if needed
        A, sign = cond(kp != k+1, do_pivot, no_pivot, (A, k, kp))

        # accumulate 2×2 pivot
        t = A[k, k+1]
        pf_A = pf_A * (sign * t)

        # rank-2 update: A += mu ⊗ nu − nu ⊗ mu
        mu = A[k, :] / t
        nu = A[:, k+1]
        A  = A + jnp.outer(mu, nu) - jnp.outer(nu, mu)

    log_pf = jnp.log(pf_A)
    return jnp.imag(log_pf), jnp.real(log_pf)

@register_pfaffian_jvp
def pfaffian_householder(A):
    """Compute the Pfaffian of a real or complex skew-symmetric matrix A (A=-A^T).
    If overwrite_a=True, the matrix A is overwritten in the process.
    This function uses the Householder tridiagonalization.
    """

    def householder_complex(x):
        """(v, tau, alpha) = householder_complex(x)
    
        Compute a Householder transformation such that
        (1 - tau * v * v^H) x = alpha * e_1
        where x and v are complex vectors, tau is 0 or 2, and
        alpha is a complex number (e_1 is the first unit vector).
        """
        
        sigma = jnp.dot(jnp.conj(x[1:]), x[1:])
        
        def true_fun(x):
            return jnp.zeros_like(x), 0, x[0]
                
        def false_fun(x):  
            norm_x = jnp.sqrt(jnp.dot(jnp.conj(x[:]), x[:]))  # Combine the calculations
            phase = jnp.exp(1j * jnp.angle(x[0]))
            v = x.at[0].set(x[0] + phase * norm_x)
            v /= jnp.linalg.norm(v)
            return v, 2, -phase * norm_x
    
        v, tau, alpha = jax.lax.cond(sigma == 0, true_fun, false_fun, x)

        return v, tau, alpha

    n = A.shape[0]
    A = jnp.asarray(A)
    pfaffian_log = 0j

    # Use a standard for loop to iterate over the range
    for i in range(n - 2):
        # Find a Householder vector to eliminate the i-th column
        v, tau, alpha = householder_complex(A[i + 1:, i])

        # Update A based on the Householder transformation
        A = A.at[i + 1, i].set(alpha)
        A = A.at[i, i + 1].set(-alpha)
        A = A.at[i + 2:, i].set(0)
        A = A.at[i, i + 2:].set(0)

        # Update the matrix block A(i+1:N,i+1:N)
        w = tau * jnp.dot(A[i + 1:, i + 1:], jnp.conjugate(v))
        A = A.at[i + 1:, i + 1:].add(jnp.outer(v, w) - jnp.outer(w, v))

        # Update pfaffian_val based on tau and the parity of i
        pfaffian_log += jnp.where(tau != 0, 1j * jnp.pi, 0j)
        pfaffian_log += jnp.where(i % 2 == 0, jnp.log(-alpha), 0j)

    pfaffian_log += jnp.log(A[n - 2, n - 1])

    return jnp.imag(pfaffian_log), jnp.real(pfaffian_log)
