import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, pmap, jacfwd, jacrev

from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental.multihost_utils import host_local_array_to_global_array

from jax.lax import fori_loop, psum, pmean
from jax.scipy.linalg import cho_factor, cho_solve
from functools import partial

from jax.flatten_util import ravel_pytree

import logging
logger = logging.getLogger()

class WavefunctionDipole(object):
    """This class serves to define a proxy of the dipole wavefunction to be
       used in the Metropolis class. 
    """

    def __init__(self, wavefunction, mesh):
        self.wavefunction = wavefunction

        self.map_axis_name = mesh.axis_names[0]

        Pspec = P(self.map_axis_name)
        Pnone = P()

        self.shard_map_logpsi = shard_map(self.vmap_logpsi, mesh=mesh,
                                            in_specs=(Pnone, Pspec, Pspec),
                                            out_specs=(Pspec),
                                            check_rep=False)

        self.shard_map_logpsi = jax.jit(self.shard_map_logpsi)

    @partial(jit, static_argnums=(0,))
    def logpsi(self, params, r, sz):
        # This function returns the log of <R|phi> = <R S| D |psi_0> with 
        # D = \sum_i P_p(i) z_i - z_CM
        rcm = jnp.mean(r, axis=0)
        proton_i = (1 + sz[:, 1]) / 2
        r_dipole = ( r - rcm[None, :] )

#        r_dipole = 2 * jnp.tanh(r_dipole / 2)

#        dipole = jnp.sum(r_dipole**2)

#        dipole = 0j
#        for i in range(3):
#            dipole += jnp.sum(r_dipole[:, i] * proton_i)

        dipole = jnp.sum(r_dipole[:, 2] * proton_i)

#        amp = 0.001

#        dipole = (jnp.exp(1j * amp * dipole) - 1) / amp / 1j 


        return jnp.log(dipole + 0j) + self.wavefunction.logpsi(params, r, sz) 
        #return self.wavefunction.logpsi(params, r, sz) 

    @partial(jit, static_argnums=(0,))
    def vmap_logpsi(self, params, r, sz):
        return vmap(self.logpsi, in_axes=(None, 0, 0))(params, r, sz) 

    @partial(jit, static_argnums=(0,))
    def psi(self, params, r, sz):
        log = self.logpsi(params, r, sz)
        return jnp.exp(log)

    @partial(jit, static_argnums=(0,))
    def vmap_psi(self, params, r, sz):
        return vmap(self.psi, in_axes=(None, 0, 0))(params, r, sz)

    @partial(jit, static_argnums=(0,))
    def flatten_params(self, parameters):
        flatten_parameters, self.unravel = ravel_pytree(parameters)
        return flatten_parameters

    @partial(jit, static_argnums=(0,))
    def unflatten_params(self, flatten_parameters):
        unflatten_parameters = self.unravel(flatten_parameters)
        return unflatten_parameters

    @partial(jit, static_argnums=(0,))
    def update_add(self, params, dparams):
        return params + dparams

class WavefunctionLIT(object):
    """This class serves to define a proxy of the wavefunction defined as (H-z)|Psi_L>
       used in the Metropolis class. 
    """

    def __init__(self, wavefunction_lit, wavefunction_dipole, mesh, params_gs):
        self.wavefunction_lit = wavefunction_lit
        self.wavefunction_dipole = wavefunction_dipole

        self.map_axis_name = mesh.axis_names[0]

        Pspec = P(self.map_axis_name)
        Pnone = P()

        self.shard_map_logpsi = shard_map(self.vmap_logpsi, mesh=mesh,
                                            in_specs=(Pnone, Pspec, Pspec),
                                            out_specs=(Pspec),
                                            check_rep=False)

        self.shard_map_logpsi = jax.jit(self.shard_map_logpsi)

        self.params_gs = params_gs

    @partial(jit, static_argnums=(0,))
    def logpsi(self, params, r, sz):
        # This function returns the log of <R|phi> = <R S| (H-z) |psi_L> with a trick. It takes the 
        # values of sig_R and sig_I from the last two elements of params.  

        #coeff = params[-1]
        #params = params[:-1]

        #logps_dipole = self.wavefunction_dipole.logpsi(self.params_gs, r, sz) 

        #logpsi_lit = self.wavefunction_dipole.logpsi(params, r, sz)

        #return jax.scipy.special.logsumexp(a=jnp.asarray([logps_dipole, logpsi_lit]), b=jnp.asarray([1, 1]) , return_sign=False)

        logpsi_lit = self.wavefunction_lit.logpsi(params, r, sz) 
        return logpsi_lit

    @partial(jit, static_argnums=(0,))
    def logpsi_new(self, params, r, sz):
        sz_pm = jnp.stack((sz, sz, sz, sz))
        r_xy = r.at[:, :2].multiply(-1)
        r_z = r.at[:, 2].multiply(-1)
        r_pm = jnp.stack((r, r_xy, r_z, -r))
        sign, log = vmap(self.wavefunction_lit.logphasepsi, in_axes=(None, 0, 0))(params, r_pm, sz_pm)
        log = log + 1j * sign
        p_sign = jnp.asarray([1, 1, -1, -1]) 
        log = jax.scipy.special.logsumexp(a=log, b=p_sign, return_sign=False)
        return log

    @partial(jit, static_argnums=(0,))
    def vmap_logpsi(self, params, r, sz):
        return vmap(self.logpsi, in_axes=(None, 0, 0))(params, r, sz) 

    @partial(jit, static_argnums=(0,))
    def psi(self, params, r, sz):
        log = self.logpsi(params, r, sz)
        return jnp.exp(log)

    @partial(jit, static_argnums=(0,))
    def vmap_psi(self, params, r, sz):
        return vmap(self.psi, in_axes=(None, 0, 0))(params, r, sz)

    @partial(jit, static_argnums=(0,))
    def flatten_params(self, parameters):
        flatten_parameters, self.unravel = ravel_pytree(parameters)
        return flatten_parameters

    @partial(jit, static_argnums=(0,))
    def unflatten_params(self, flatten_parameters):
        unflatten_parameters = self.unravel(flatten_parameters)
        return unflatten_parameters

    @partial(jit, static_argnums=(0,))
    def update_add(self, params, dparams):
        return params + dparams

class NeuraLIT(object):

    def __init__(self, wavefunction_dipole, wavefunction_lit, observables_lit, energy_gs, mesh, config, nparams_lit):

        # prep shardmap
        self.map_axis_name = mesh.axis_names[0]

        # Shmap functions
        Pspec = P(self.map_axis_name)
        Pnone = P()

        self.shard_map_sr_overlap = shard_map(self.sr_overlap, mesh=mesh,
                                            in_specs=(Pnone, Pnone, Pnone, Pspec, Pspec, Pspec, Pspec, Pnone, Pnone, Pnone, Pnone),
                                            out_specs=(Pnone),
                                            check_rep=False)

        self.shard_map_sr_overlap = jax.jit(self.shard_map_sr_overlap)

        self.wavefunction_dipole = wavefunction_dipole
        self.wavefunction_lit = wavefunction_lit
        self.observables_lit  = observables_lit

        self.delta = config.delta
        self.energy_gs = energy_gs
        self.eps = config.eps

        self.alpha = 0.9
        self.beta = 0.99

        self.nparams_lit = nparams_lit

        self.iterative = config.iterative

        if self.iterative == 'jax.scipy':
           cg_solver = jax.scipy.sparse.linalg.cg
        elif self.iterative == 'solve_while_loop':
           cg_diag = jnp.ones(self.nparams_lit)
           def cg_solver(A, b, x0=None, tol=1e-5, atol=0, maxiter=1000):
               return cg_solve_while_loop(A, b, x0, cg_diag, self.map_axis_name, tol=tol, atol=atol, maxiter=maxiter)
        elif self.iterative == 'solve_fori_loop':
           cg_diag = jnp.ones(self.nparams_lit)
           def cg_solver(A, b, x0=None, tol=1e-5, atol=0, maxiter=1000):
               return cg_solve_fori_loop(A, b, x0=x0, A_diag=cg_diag, tol=tol, atol=atol, maxiter=maxiter)
        else:
           logger.error('Invalid iterative method. Please choose between "jax.scipy", "solve_while_loop", or "solve_fori_loop".')
           raise ValueError('Invalid iterative method. Please choose between "jax.scipy", "solve_while_loop", or "solve_fori_loop".')

        self.cg_solver = cg_solver


    @partial(jit, static_argnums=(0,))
    def elocal(self, params, r, sz, sig_R, sig_I):
        """ This function computes <R, S | (H - z) |psi_L> / <R, S  |psi_L>
        """
        kinetic, _ = self.observables_lit.kinetic_energy(params, r, sz)
        potential = self.observables_lit.potential_energy(params, r, sz)

        return kinetic + potential - self.energy_gs - sig_R - 1j * sig_I

    @partial(jit, static_argnums=(0,))
    def vmap_elocal(self, params, r, sz, sig_R, sig_I):
        return vmap(self.elocal, in_axes=(None, 0, 0, None, None))(params, r, sz, sig_R, sig_I) 

    @partial(jit, static_argnums=(0,))
    def d_elocal(self, params, r, sz, sig_R, sig_I):
        elocal_r = lambda params: jnp.real(self.elocal(params, r, sz, sig_R, sig_I))
        elocal_i = lambda params: jnp.imag(self.elocal(params, r, sz, sig_R, sig_I))

        d_elocal_r = self.wavefunction_lit.flatten_params(jax.grad(elocal_r)(params))
        d_elocal_i = self.wavefunction_lit.flatten_params(jax.grad(elocal_i)(params))
        d_elocal = d_elocal_r + 1j * d_elocal_i
        return d_elocal

    @partial(jit, static_argnums=(0,))
    def vmap_d_elocal(self, params, r, sz, sig_R, sig_I):
        return vmap(self.d_elocal, in_axes=(None, 0, 0, None, None))(params, r, sz, sig_R, sig_I)

    @partial(jit, static_argnums=(0,))
    def d_logpsi_lit(self, params, r, sz):
        
        logpsi_r = lambda params: jnp.real(self.wavefunction_lit.logpsi(params, r, sz))
        logpsi_i = lambda params: jnp.imag(self.wavefunction_lit.logpsi(params, r, sz))

        d_logpsi_r = self.wavefunction_lit.flatten_params(jax.grad(logpsi_r)(params))
        d_logpsi_i = self.wavefunction_lit.flatten_params(jax.grad(logpsi_i)(params))
        
        d_logpsi = d_logpsi_r + 1j * d_logpsi_i
        return d_logpsi   
    
    @partial(jit, static_argnums=(0,))
    def vmap_d_logpsi_lit(self, params, r, sz):
        return vmap(self.d_logpsi_lit, in_axes=(None, 0, 0))(params, r, sz)

    @partial(jit, static_argnums=(0,))
    def fidelity_compute_rhs_test(self, params_lit, params_gs, r_d, sz_d, sig_R, sig_I):
        """
        Computes the quantum fidelity between
        ( H - E_0 - sig_R - i sig_I) |Psi_L> 
        and 
        |Psi_D> = D |Psi_0>
        as well as its derivatives sampling from |Psi_D|^2
        """     
        logpsi_dipole =  self.wavefunction_dipole.vmap_logpsi(params_gs, r_d, sz_d)         
        logpsi_lit = self.wavefunction_lit.vmap_logpsi(params_lit, r_d, sz_d)
        elocal_lit = self.vmap_elocal(params_lit, r_d, sz_d, sig_R, sig_I)

        exp_diff = jnp.exp(logpsi_lit - logpsi_dipole)
        abs_exp_diff_squared = jnp.conjugate(exp_diff) * exp_diff
        
        n_2 = elocal_lit * exp_diff    
        n_2 = pmean(jnp.mean(n_2), axis_name=self.map_axis_name)

        n_1 = jnp.conjugate(n_2)
        d_2 = jnp.conjugate(elocal_lit) * elocal_lit * abs_exp_diff_squared
        d_2 = pmean(jnp.mean(d_2), axis_name=self.map_axis_name)

        overlap = n_1 * n_2 / d_2
        
        return jnp.real(overlap)

    @partial(jit, static_argnums=(0,))
    def fidelity_compute_rhs(self, params_lit, params_gs, r_l, sz_l, r_d, sz_d, sig_R, sig_I):
        """
        Computes the quantum fidelity between
        ( H - E_0 - sig_R - i sig_I) |Psi_L> 
        and 
        |Psi_D> = D |Psi_0>
        as well as its derivatives sampling from |Psi_D|^2
        """     
        logpsi_dipole =  self.wavefunction_dipole.vmap_logpsi(params_gs, r_d, sz_d)         
        logpsi_lit = self.wavefunction_lit.vmap_logpsi(params_lit, r_d, sz_d)
        d_logpsi_lit = self.vmap_d_logpsi_lit(params_lit, r_d, sz_d)
        elocal_lit = self.vmap_elocal(params_lit, r_d, sz_d, sig_R, sig_I)
        d_elocal_lit = self.vmap_d_elocal(params_lit, r_d, sz_d, sig_R, sig_I)

        exp_diff = jnp.exp(logpsi_lit - logpsi_dipole)
        abs_exp_diff_squared = jnp.conjugate(exp_diff) * exp_diff
        
        n_2 = elocal_lit * exp_diff  
        
        weight = jnp.abs(n_2)**2

        weight = weight / psum(jnp.sum(weight), axis_name=self.map_axis_name)

        ESS = 1.0 / psum(jnp.sum(weight**2), axis_name=self.map_axis_name)

        jax.debug.print("ESS: {}", ESS)

        n_2 = pmean(jnp.mean(n_2), axis_name=self.map_axis_name)

        d_n_2 = (d_elocal_lit + elocal_lit[:, None] * d_logpsi_lit) * exp_diff[:, None]

        jac = d_elocal_lit / elocal_lit[:, None] + d_logpsi_lit
        jac = jac - psum(jnp.sum(jac * weight[:, None], axis=0), axis_name=self.map_axis_name)

        d_n_2 = pmean(jnp.mean(d_n_2, axis=0), axis_name=self.map_axis_name)
    
        n_1 = jnp.conjugate(n_2)
        d_n_1 = jnp.conjugate(d_n_2)
               
        d_2 = jnp.conjugate(elocal_lit) * elocal_lit * abs_exp_diff_squared
        d_2 = pmean(jnp.mean(d_2), axis_name=self.map_axis_name)
        d_d_2 = jnp.conjugate(elocal_lit[:, None]) * (d_elocal_lit + elocal_lit[:, None] * d_logpsi_lit) * abs_exp_diff_squared[:, None]
        d_d_2 = pmean(jnp.mean(2 * jnp.real(d_d_2), axis = 0), axis_name=self.map_axis_name)

        overlap_rhs = n_1 * n_2 / d_2
        d_overlap_rhs = overlap_rhs * (d_n_1 / n_1 + d_n_2 / n_2  - d_d_2 / d_2 )
        overlap_lhs = 0 + 0j
        overlap = 0 + 0j
        return jnp.real(overlap_rhs), jnp.real(d_overlap_rhs), weight, jac


    @partial(jit, static_argnums=(0,))
    def fidelity_compute_filippo_rw(self, params_lit, params_gs, r_l, sz_l, r_d, sz_d, sig_R, sig_I):

        logpsi_dipole_l =  self.wavefunction_dipole.vmap_logpsi(params_gs, r_l, sz_l)        
        logpsi_lit_l = self.wavefunction_lit.vmap_logpsi(params_lit, r_l, sz_l)
        d_logpsi_lit_l = self.vmap_d_logpsi_lit(params_lit, r_l, sz_l)
        elocal_lit_l = self.vmap_elocal(params_lit, r_l, sz_l, sig_R, sig_I)
        d_elocal_lit_l = self.vmap_d_elocal(params_lit, r_l, sz_l, sig_R, sig_I)

        logpsi_dipole_d =  self.wavefunction_dipole.vmap_logpsi(params_gs, r_d, sz_d)        
        logpsi_lit_d = self.wavefunction_lit.vmap_logpsi(params_lit, r_d, sz_d)
        elocal_lit_d = self.vmap_elocal(params_lit, r_d, sz_d, sig_R, sig_I)

        F_d = elocal_lit_d * jnp.exp(logpsi_lit_d - logpsi_dipole_d)
        F_l = elocal_lit_l * jnp.exp(logpsi_lit_l - logpsi_dipole_l)
    
        H_loc = pmean(jnp.mean(F_d), axis_name=self.map_axis_name) / F_l
        
        weight = jnp.abs(F_l)**2 
        weight = weight / psum(jnp.sum(weight), axis_name=self.map_axis_name)
        N_tot = psum(jnp.array(weight.shape[0], weight.dtype), axis_name=self.map_axis_name)
        ess   = 1.0 / psum(jnp.sum(weight**2), axis_name=self.map_axis_name)  # if weight normalized
        ess = ess / N_tot

        fidelity = psum(jnp.sum(H_loc * weight), axis_name=self.map_axis_name) 

        J_l = d_elocal_lit_l / elocal_lit_l[:, None] + d_logpsi_lit_l
        J_l = J_l - psum(jnp.sum(J_l * weight[:, None], axis=0), axis_name=self.map_axis_name) 
        
        d_fidelity = 2 * jnp.real(J_l * jnp.conjugate(H_loc[:, None]))
        d_fidelity = psum(jnp.sum(d_fidelity * weight[:, None], axis = 0), axis_name=self.map_axis_name) 

        return jnp.real(fidelity), d_fidelity, weight, J_l, ess

    @partial(jit, static_argnums=(0,))
    def fidelity_compute_chatgpt_kl(self, params_lit, params_gs, r_l, sz_l, r_d, sz_d, sig_R, sig_I):

        # --- "l" batch (denominator & grad) ---
        logpsi_dipole_l =  self.wavefunction_dipole.vmap_logpsi(params_gs, r_l, sz_l)
        logpsi_lit_l    = self.wavefunction_lit.vmap_logpsi(params_lit, r_l, sz_l)
        d_logpsi_lit_l  = self.vmap_d_logpsi_lit(params_lit, r_l, sz_l)
        elocal_lit_l    = self.vmap_elocal(params_lit, r_l, sz_l, sig_R, sig_I)
        d_elocal_lit_l  = self.vmap_d_elocal(params_lit, r_l, sz_l, sig_R, sig_I)
 
        # --- "d" batch (numerator) ---
        logpsi_dipole_d = self.wavefunction_dipole.vmap_logpsi(params_gs, r_d, sz_d)
        logpsi_lit_d    = self.wavefunction_lit.vmap_logpsi(params_lit, r_d, sz_d)
        elocal_lit_d    = self.vmap_elocal(params_lit, r_d, sz_d, sig_R, sig_I)
 
        # Reuse exp(logpsi_L - logpsi_d)
        exp_diff_l = jnp.exp(logpsi_lit_l - logpsi_dipole_l)
        exp_diff_d = jnp.exp(logpsi_lit_d - logpsi_dipole_d)
 
        F_l = elocal_lit_l * exp_diff_l
        F_d = elocal_lit_d * exp_diff_d
 
        # Keep H_loc
        H_loc = pmean(jnp.mean(F_d), axis_name=self.map_axis_name) / F_l
 
        # ---------- weights & global stats ----------
        weight = jnp.abs(F_l)**2
        w_sum  = psum(jnp.sum(weight),    axis_name=self.map_axis_name)
        w2_sum = psum(jnp.sum(weight**2), axis_name=self.map_axis_name)
        N_tot  = psum(jnp.array(weight.shape[0], weight.dtype), axis_name=self.map_axis_name)
        Z      = w_sum / N_tot  # E_q[w]
 
        # Shared gradient pieces: dF and 2 Re(F* dF)
        dF_l       = (d_elocal_lit_l + elocal_lit_l[:, None] * d_logpsi_lit_l) * exp_diff_l[:, None]
        twoRe_FdF  = 2.0 * jnp.real(jnp.conjugate(F_l)[:, None] * dF_l)  # (N,P)
        dZ         = psum(jnp.sum(twoRe_FdF, axis=0), axis_name=self.map_axis_name) / N_tot
 
        # ---------- KL gradient: forward vs reverse (simple if) ----------
        use_reverse_kl = getattr(self, "use_reverse_kl", True)
 
        if use_reverse_kl:
            # Reverse KL: KL(p||q) = E_q[w log w]/Z - log Z  (no eps, no 1/w)
            w_logw_local = jnp.where(weight > 0, weight * jnp.log(weight), 0.0)
            A = psum(jnp.sum(w_logw_local), axis_name=self.map_axis_name) / N_tot
            logw_plus1 = jnp.where(weight > 0, jnp.log(weight) + 1.0, 1.0)
            dA = psum(jnp.sum(logw_plus1[:, None] * twoRe_FdF, axis=0),
                      axis_name=self.map_axis_name) / N_tot
            d_KL = (dA / Z) - ((A + Z) / (Z**2)) * dZ
        else:
            # Forward KL: KL(q||p) = log Z - E_q[log w]  (robust to big weights)
            # Safe reciprocal; avoid 1/0 (JAX evaluates both branches of where)
            eps = jnp.asarray(jnp.finfo(weight.dtype).tiny, weight.dtype)
            inv_w  = 1.0 / jnp.maximum(weight, eps)
            dElogw = psum(jnp.sum(inv_w[:, None] * twoRe_FdF, axis=0),
                          axis_name=self.map_axis_name) / N_tot
            d_KL = (dZ / Z) - dElogw
        # ---------------------------------------------------------------
 
        # Normalize weights for H_loc estimator + diagnostics (reuse w_sum)
        weight = weight / w_sum
        ess    = (w_sum**2 / w2_sum) / N_tot

        tiny = jnp.asarray(jnp.finfo(weight.dtype).tiny, weight.dtype)
        u = F_l / jnp.sqrt(jnp.abs(F_l)**2 + tiny)
        R_phi = jnp.abs(psum(jnp.sum(weight * u), axis_name=self.map_axis_name))
        jax.debug.print("R_phi={:.3f}", R_phi)

        # Fidelity
        fidelity = psum(jnp.sum(H_loc * weight), axis_name=self.map_axis_name)
 
        # J_l centering under weights
        J_l = d_elocal_lit_l / elocal_lit_l[:, None] + d_logpsi_lit_l
        J_l = J_l - psum(jnp.sum(J_l * weight[:, None], axis=0), axis_name=self.map_axis_name)
 
        d_fidelity = 2 * jnp.real(J_l * jnp.conjugate(H_loc[:, None]))
        d_fidelity = psum(jnp.sum(d_fidelity * weight[:, None], axis=0), axis_name=self.map_axis_name)
 
        # Combine: ascent on fidelity, descent on KL (tune lambda_amp small)
        lambda_amp = getattr(self, "lambda_amp", 2.)
        d_fidelity = d_fidelity - lambda_amp * d_KL
        #d_fidelity = - lambda_amp * d_KL
 
        return jnp.real(fidelity), d_fidelity, weight, J_l, ess

    @partial(jit, static_argnums=(0,))
    def fidelity_compute_chatgpt(self, params_lit, params_gs, r_l, sz_l, r_d, sz_d, sig_R, sig_I):

        # --- "l" batch (denominator & grad) ---
        logpsi_dipole_l =  self.wavefunction_dipole.vmap_logpsi(params_gs, r_l, sz_l)
        logpsi_lit_l    = self.wavefunction_lit.vmap_logpsi(params_lit, r_l, sz_l)
        d_logpsi_lit_l  = self.vmap_d_logpsi_lit(params_lit, r_l, sz_l)
        elocal_lit_l    = self.vmap_elocal(params_lit, r_l, sz_l, sig_R, sig_I)
        d_elocal_lit_l  = self.vmap_d_elocal(params_lit, r_l, sz_l, sig_R, sig_I)
 
        # --- "d" batch (numerator) ---
        logpsi_dipole_d = self.wavefunction_dipole.vmap_logpsi(params_gs, r_d, sz_d)
        logpsi_lit_d    = self.wavefunction_lit.vmap_logpsi(params_lit, r_d, sz_d)
        elocal_lit_d    = self.vmap_elocal(params_lit, r_d, sz_d, sig_R, sig_I)
 
        # Reuse exp(logpsi_L - logpsi_d) once
        exp_diff_l = jnp.exp(logpsi_lit_l - logpsi_dipole_l)
        exp_diff_d = jnp.exp(logpsi_lit_d - logpsi_dipole_d)
 
        F_d = elocal_lit_d * exp_diff_d
        F_l = elocal_lit_l * exp_diff_l
 
        # Keep H_loc
        H_loc = pmean(jnp.mean(F_d), axis_name=self.map_axis_name) / F_l
 
        # ---------- Neyman χ² (scale-invariant) ----------
        weight = jnp.abs(F_l)**2  # |F_l|^2
 
        # Global aggregates (reused below)
        w_sum  = psum(jnp.sum(weight),   axis_name=self.map_axis_name)
        w2_sum = psum(jnp.sum(weight**2), axis_name=self.map_axis_name)
        N_tot  = psum(jnp.array(weight.shape[0], weight.dtype), axis_name=self.map_axis_name)
 
        # Moments under q = |psi_d|^2
        Z  = w_sum  / N_tot  # E[|F|^2]
        M4 = w2_sum / N_tot  # E[|F|^4]
 
        # dF = (d_eloc + eloc * dlogpsi) * exp(...)
        dF_l     = (d_elocal_lit_l + elocal_lit_l[:, None] * d_logpsi_lit_l) * exp_diff_l[:, None]
        F_l_conj = jnp.conjugate(F_l)
 
        # E[F* dF] and E[|F|^2 F* dF] (one reduction each)
        Tz_sum  = psum(jnp.sum(F_l_conj[:, None] * dF_l, axis=0), axis_name=self.map_axis_name)
        Tm4_sum = psum(jnp.sum((weight[:, None] * F_l_conj[:, None]) * dF_l, axis=0), axis_name=self.map_axis_name)
        Tz  = Tz_sum  / N_tot
        Tm4 = Tm4_sum / N_tot
 
        dZ      = 2.0 * jnp.real(Tz)
        dM4     = 4.0 * jnp.real(Tm4)
        d_L_amp = (dM4 / (Z**2)) - (2.0 * M4 / (Z**3)) * dZ
        # -----------------------------------------------
 
        # Normalize weights for H_loc estimator + diagnostics (reuse w_sum)
        weight = weight / w_sum
        ess    = (w_sum**2 / w2_sum) / N_tot
 
        # Fidelity
        fidelity = psum(jnp.sum(H_loc * weight), axis_name=self.map_axis_name)
 
        # J_l centering under weights (kept as-is)
        J_l = d_elocal_lit_l / elocal_lit_l[:, None] + d_logpsi_lit_l
        J_l = J_l - psum(jnp.sum(J_l * weight[:, None], axis=0), axis_name=self.map_axis_name)
 
        d_fidelity = 2 * jnp.real(J_l * jnp.conjugate(H_loc[:, None]))
        d_fidelity = psum(jnp.sum(d_fidelity * weight[:, None], axis=0), axis_name=self.map_axis_name)
 
        # Merge amplitude penalty gradient (least invasive)
        lambda_amp = getattr(self, "lambda_amp", 0.1)  # set >0 to enable
        #d_fidelity = - lambda_amp * d_L_amp
        d_fidelity = d_fidelity - lambda_amp * d_L_amp
 
        return jnp.real(fidelity), d_fidelity, weight, J_l, ess

    def sr_overlap(self, itr, params_lit, params_gs, r_l, sz_l, r_d, sz_d, m_overlap, g2_overlap, sig_R, sig_I):
        itr = itr + 1

#        overlap, d_overlap, weight, jac = self.fidelity_compute_rhs(params_lit, params_gs, r_l, sz_l, r_d, sz_d, sig_R, sig_I)

        #overlap, d_overlap, weight, jac, ess = self.fidelity_compute_filippo_rw(params_lit, params_gs, r_l, sz_l, r_d, sz_d, sig_R, sig_I)
        #overlap, d_overlap, weight, jac, ess = self.fidelity_compute_chatgpt(params_lit, params_gs, r_l, sz_l, r_d, sz_d, sig_R, sig_I)
        overlap, d_overlap, weight, jac, ess = self.fidelity_compute_chatgpt_kl(params_lit, params_gs, r_l, sz_l, r_d, sz_d, sig_R, sig_I)
        overlap_rhs = 0.
        overlap_lhs = 0.

        def sr_update(itr, weight, jac, d_overlap, m_overlap):
            # --- Fisher metric S = Re[J^H W J] aggregated across devices ---
            Jr, Ji = jnp.real(jac), jnp.imag(jac)
            WJr = weight[:, None] * Jr
            WJi = weight[:, None] * Ji
            S_local = Jr.T @ WJr + Ji.T @ WJi         # real (P,P)
            S = psum(S_local, axis_name=self.map_axis_name)

            # --- scale-invariant damping: λ = eps * mean(diag(S)) ---
            mean_diag = jnp.mean(jnp.diag(S))
            jax.debug.print("mean_diag: {}", mean_diag)
            tiny = jnp.asarray(jnp.finfo(d_overlap.dtype).eps, d_overlap.dtype)
            lam = self.eps * (mean_diag + tiny)

            idx = jnp.diag_indices(S.shape[0])
            S = S.at[idx].add(lam)

            mu = getattr(self, "spring_mu", 0.99)
            rhs = d_overlap + (lam * mu) * m_overlap

        # ---- solve (S + λ I) δ = rhs  (full Cholesky)
            U, low = cho_factor(S, lower=True, check_finite=False)
            d_overlap = cho_solve((U, low), rhs)
            m_overlap = d_overlap
            return d_overlap, m_overlap

        def adam_update(itr, d_overlap, m_overlap, g2_overlap):
            m_overlap = self.alpha * m_overlap + (1. - self.alpha) * d_overlap
            mh_overlap = m_overlap / ( 1. - self.alpha**itr)       
            g2_overlap = self.beta * g2_overlap + (1. - self.beta) * d_overlap**2
            g2h_overlap = g2_overlap / ( 1. - self.beta**itr)
            d_overlap = mh_overlap / ( jnp.sqrt(g2h_overlap) + 1e-8 )
            return d_overlap, m_overlap, g2_overlap 
#ALE
#        d_overlap, m_overlap, g2_overlap = adam_update(itr, d_overlap, m_overlap, g2_overlap)
        d_overlap, m_overlap = sr_update(itr, weight, jac, d_overlap, m_overlap)
        d_overlap = self.wavefunction_lit.unflatten_params(self.delta * d_overlap) 

        d_overlap = pmean(d_overlap, axis_name=self.map_axis_name)
        m_overlap = pmean(m_overlap, axis_name=self.map_axis_name)
        g2_overlap = pmean(g2_overlap, axis_name=self.map_axis_name)
        ess = pmean(ess, axis_name=self.map_axis_name)

        #params_test = jax.tree_util.tree_map(self.wavefunction_lit.update_add, params_lit, d_overlap)
        #overlap_test = self.fidelity_compute_rhs_test(params_test, params_gs, r_d, sz_d, sig_R, sig_I)

        overlap_test = 0.

        return (overlap, overlap_lhs, overlap_rhs, overlap_test, d_overlap, m_overlap, g2_overlap, ess)

    @partial(jit, static_argnums=(0,))
    def lit_compute_xilin(self, params, r, sz, logpsi_dipole, logpsi_gs, sig_R, sig_I, sum_rule):

        logpsi_lit = self.wavefunction_lit.vmap_logpsi(params, r, sz)
        elocal_lit = self.vmap_elocal(params, r, sz, sig_R, sig_I)

        logpsi_lhs = jnp.log(elocal_lit) + logpsi_lit

        phase_lit = 1 / jnp.mean(jnp.exp(logpsi_lhs - logpsi_dipole))
        logpsi_lit = self.wavefunction_lit.vmap_logpsi(params, r, sz) + jnp.log(phase_lit)
        numerator = jnp.mean(jnp.exp(logpsi_lit - logpsi_dipole))
        denominator = jnp.mean(jnp.abs(jnp.exp(2 * (logpsi_gs - logpsi_dipole) ) ) )

        lit_xilin = jnp.imag(numerator) / sig_I / denominator

        # error bound
        exp_delta = jnp.exp(logpsi_lit - logpsi_dipole)

        # ⟨ψ_D|ψ_L⟩ / ⟨ψ_D|ψ_D⟩
        ratio = jnp.mean(exp_delta)

        # norm = ⟨ψ_L|ψ_L⟩ / ⟨ψ_D|ψ_D⟩
        norm = jnp.mean(jnp.abs(exp_delta)**2)

        # (1 − |⟨ψ_D|ψ_L⟩|²/⟨ψ_L|ψ_L⟩)
        overlap = jnp.abs(ratio)**2 / norm
        removal_fac  = jnp.sqrt(1 - overlap)

        # fidelity calculation
        F = elocal_lit * exp_delta     
        H_loc = jnp.mean(F) / F                  
        weights = jnp.abs(F)**2
        fidelity = jnp.mean(H_loc * weights) / jnp.mean(weights)

        # logging
        logger.info(f"sig_R: {sig_R}, fidelity: {fidelity}, lit: {lit_xilin}, removal: {removal_fac}")

        # error‐bound formulas
        primary_bound   = (jnp.sqrt(sum_rule) / sig_I
                           * jnp.sqrt(lit_xilin)
                           * jnp.sqrt((1 - fidelity) / fidelity))
        secondary_bound = removal_fac * primary_bound

        primary_bound = jnp.real(primary_bound)
        secondary_bound = jnp.real(secondary_bound)
        
        error_bound = jnp.minimum(primary_bound, secondary_bound)

        return lit_xilin, error_bound

    def lit_error_bound(self, params, r, sz, logpsi_dipole, sig_R, sig_I, sum_rule, lit):
        # compute log‐amplitudes and local energy
        logpsi_lit   = self.wavefunction_lit.vmap_logpsi(params, r, sz)
        elocal_lit   = self.vmap_elocal(params, r, sz, sig_R, sig_I)

        # build the core overlap factor Δ = logψ_L − logψ_D and its exp
        exp_delta    = jnp.exp(logpsi_lit - logpsi_dipole)

        # ratio = ⟨ψ_D|ψ_L⟩ / ⟨ψ_D|ψ_D⟩
        ratio        = jnp.mean(exp_delta)

        # norm = ⟨ψ_L|ψ_L⟩ / ⟨ψ_D|ψ_D⟩
        norm         = jnp.mean(jnp.abs(exp_delta)**2)

        # removal factor = √(1 − |⟨ψ_D|ψ_L⟩|²/⟨ψ_L|ψ_L⟩)
        overlap      = jnp.abs(ratio)**2 / norm
        removal_fac  = jnp.sqrt(1 - overlap)

        # fidelity calculation
        F            = elocal_lit * exp_delta     # unnormalized weights
        avg_F        = jnp.mean(F)
        H_loc        = avg_F / F                  # local H_rel factor per sample
        weights      = jnp.abs(F)**2
        fidelity     = jnp.mean(H_loc * weights) / jnp.mean(weights)

        # logging
        logger.info(f"sig_R: {sig_R}, fidelity: {fidelity}, lit: {lit}, removal: {removal_fac}")

        # error‐bound formulas
        primary_bound   = (jnp.sqrt(sum_rule) / sig_I
                           * jnp.sqrt(lit)
                           * jnp.sqrt((1 - fidelity) / fidelity))
        secondary_bound = removal_fac * primary_bound

        # return real parts in case of vanishing imaginary noise
        return jnp.real(primary_bound), jnp.real(secondary_bound)

