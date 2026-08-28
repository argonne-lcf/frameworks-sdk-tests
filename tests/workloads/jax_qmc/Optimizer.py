import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, pmap, jacfwd, jacrev, lax

from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental.multihost_utils import host_local_array_to_global_array

from jax.lax import psum, pmean
from jax.scipy.linalg import cho_factor, cho_solve
from custom_cg_solvers import cg_solve_fori_loop, cg_solve_while_loop
from functools import partial

import logging
logger = logging.getLogger()


@jit
def conj_transpose(M):
    return jnp.conjugate(jnp.transpose(M))


class Optimizer(object):

    def __init__(self, n_devices, nparams, wavefunction, observables, config, mesh):
        self.lbd = config.lbd
        self.alpha = config.alpha
        self.beta = config.beta
        self.alpha_spring = config.alpha_spring
        self.n_devices = n_devices
        self.eps = config.eps
        self.delta = config.delta
        self.nparams = nparams
        self.nwalk = config.nwalk
        self.npart = config.npart
        self.ndim = config.ndim
        self.nav = config.nav
        self.nac = config.nac
        self.wavefunction = wavefunction
        self.observables = observables
        self.solver = config.solver
        self.iterative = config.iterative
        self.remove_cm = config.remove_cm
        self.itr = 0
        self.dp_i = jnp.zeros(self.nparams)
        self.g2_i = jnp.zeros(self.nparams)
        self.m_i = jnp.zeros(self.nparams)
        self.nstep = 1
        self.getder = jax.jit(self.getder)
        self.dist = jax.jit(self.dist)

        # prep shardmap
        self.map_axis_name = mesh.axis_names[0]

        # Replicate some data over devices
        self.dp_i = host_local_array_to_global_array(self.dp_i, mesh, P())
        self.g2_i = host_local_array_to_global_array(self.g2_i, mesh, P())
        self.m_i = host_local_array_to_global_array(self.m_i, mesh, P())

        # Selecting solvers
        if self.solver == 'Adam':
            self.sr_solver = self.adam
#           self.sr_solver = self.adam_variance
            
        elif self.solver == 'Cholesky':
            self.sr_solver = self.sr_cholesky
#           self.sr_solver = self.sr_cholesky_variance
            
        elif self.solver == 'Pseudo':
            self.sr_solver = self.sr_pseudo
        elif self.solver == 'CG':
            self.sr_solver = self.sr_cg

            if self.iterative == 'jax.scipy':
                cg_solver = jax.scipy.sparse.linalg.cg
            elif self.iterative == 'solve_while_loop':
                cg_diag = jnp.ones(self.nparams)

                def cg_solver(A, b, x0=None, tol=1e-5, atol=0, maxiter=1000):
                    return cg_solve_while_loop(A, b, x0, cg_diag, self.map_axis_name, tol=tol, atol=atol, maxiter=maxiter)
            
            elif self.iterative == 'solve_fori_loop':
                cg_diag = jnp.ones(self.nparams)

                def cg_solver(A, b, x0=None, tol=1e-5, atol=0, maxiter=1000):
                    return cg_solve_fori_loop(A, b, x0=x0, A_diag=cg_diag, tol=tol, atol=atol, maxiter=maxiter)

            else:
                logger.error('Invalid iterative method. Please choose between "jax.scipy", "solve_while_loop", or "solve_fori_loop".')
                raise ValueError('Invalid iterative method. Please choose between "jax.scipy", "solve_while_loop", or "solve_fori_loop".')

            self.cg_solver = cg_solver
        else:
            logger.error('Invalid solver. Please choose between "Adam", "Cholesky", "Pseudo", or "CG".')
            raise ValueError('Invalid solver. Please choose between "Adam", "Cholesky", "Pseudo", or "CG".')

        # Shmap functions
        Pspec = P(self.map_axis_name)
        Pnone = P()

        self.shard_map_optimize = shard_map(self.optimize, mesh=mesh,
                                            in_specs=(Pnone, Pspec, Pspec, Pspec, Pnone),
                                            out_specs=(Pnone),
                                            check_rep=False)
        self.shard_map_optimize = jax.jit(self.shard_map_optimize)

    def getder(self, params, r, sz):
        logpsi_r = lambda params: jnp.real(self.wavefunction.logpsi(params, r, sz))
        logpsi_i = lambda params: jnp.imag(self.wavefunction.logpsi(params, r, sz))

        dlogpsi_r = self.wavefunction.flatten_params(jax.grad(logpsi_r)(params))
        dlogpsi_i = self.wavefunction.flatten_params(jax.grad(logpsi_i)(params))

        dlogpsi = dlogpsi_r + 1j * dlogpsi_i
        return dlogpsi

    def vmap_getder(self, params, r, sz):
        jac = vmap(self.getder, in_axes=(None, 0, 0), out_axes=(0))(params, r, sz)
        jac = jac - pmean(jnp.mean(jac, axis=0), axis_name=self.map_axis_name) 
        return jac

    @partial(jit, static_argnums=(0,))
    def getder_elocal(self, params, r, sz):      
        elocal_r = lambda params: jnp.real(self.observables.elocal(params, r, sz)[0])
        elocal_i = lambda params: jnp.imag(self.observables.elocal(params, r, sz)[0])

        d_elocal_r = self.wavefunction.flatten_params(jax.grad(elocal_r)(params))
        d_elocal_i = self.wavefunction.flatten_params(jax.grad(elocal_i)(params))
        
        d_elocal = d_elocal_r + 1j * d_elocal_i
        return d_elocal

    def vmap_getder_elocal(self, params, r, sz):
        nwalk_local = self.nwalk // self.n_devices

        r_rs  = r.reshape(self.nav, nwalk_local, *r.shape[1:])
        sz_rs = sz.reshape(self.nav, nwalk_local, *sz.shape[1:])

        inner = lambda r_nav, sz_nav: vmap(self.getder_elocal, in_axes=(None, 0, 0), out_axes=0)(params, r_nav, sz_nav)

        out = lax.map(lambda pair: inner(*pair), (r_rs, sz_rs))

        batch_local = self.nav * nwalk_local
        return out.reshape(batch_local, *out.shape[2:])

    @partial(jax.jit, static_argnums=(0,))
    def sr_cg(self, params, r, sz, energy, dp0_i, g2_i, m_i, itr):
        """Parameters' update according to the SR algorithm with Conjugate Gradient solver
        Args:
        params: initial variational parameters
        r: array with shape (nwalk/n_devices * nav, npart, ndim)
        sz: array with shape (nwalk/n_devices * nav, npart, 2)
        energy: array with shape (nwalk/n_devices * nav)
        dp0_i: array with shape (nparams) representing the initial guess of the CG solver
        g2_i: accumulated second order derivative squared  (nparams)
        itr : iteration

        Returns:
        dp_i: array with the same shape as ``params`` representing the best parameters' update (nparams)
        g2_i: updated accumulated second order derivative squared (nparams)
        """
        energy = energy - pmean(jnp.mean(energy), axis_name=self.map_axis_name)

        nsamples = self.nwalk * self.nav

        jac = self.vmap_getder(params, r, sz)
        jac_conjugate = jnp.conjugate(jac)

        f_i = jnp.real(-2 * psum(jnp.matmul(energy, jac_conjugate), axis_name=self.map_axis_name) / nsamples)
        g2_i = self.beta * g2_i + (1. - self.beta) * f_i**2
        g2h_i = g2_i / (1. - self.beta**itr)

        def cg_mult(v_i):
            # energy.shape[0] = nsamples/ndevices; pmean divides by ndevices
            # Overall the first term is divided by nsamples as required.
            local_mult = jnp.matmul(jnp.matmul(jac, v_i), jac_conjugate) / energy.shape[0] + self.eps * (0.01 + jnp.sqrt(g2h_i)) * v_i
            global_mult = pmean(local_mult, axis_name=self.map_axis_name) 
            return jnp.real(global_mult)

        # Iterative method selection
        dp_i, info = self.cg_solver(cg_mult, f_i, x0=dp0_i, tol=1e-5, atol=0.0, maxiter=200)

        dp_i = dp_i - self.lbd * self.wavefunction.flatten_params(params)
        return dp_i, g2_i, m_i

    def sr_cholesky(self, params, r, sz, energy, dp0_i, g2_i, m_i, itr):
        """Parameters' update according to the SR algorithm with Cholesky solver
        Args:
        params: initial variational parameters
        r: array with shape (nwalk/n_devices * nav, npart, ndim)
        sz: array with shape (nwalk/n_devices * nav, npart, 2)
        energy: array with shape (nwalk/n_devices * nav)
        g2_i: accumulated second order derivative squared  (nparams)
        itr : iteration

        Returns:
        dp_i: array with the same shape as ``params`` representing the best parameters' update (nparams)
        g2_i: updated accumulated second order derivative squared (nparams)
        """
        energy_mean = pmean(jnp.mean(energy), axis_name=self.map_axis_name)
        energy2_mean = pmean(jnp.mean(jnp.conjugate(energy)*energy), axis_name=self.map_axis_name)

        test_variance = jnp.real( energy2_mean - jnp.abs(energy_mean)**2 )

        jax.debug.print("test_variance: {}", test_variance)

        energy = energy - pmean(jnp.mean(energy), axis_name=self.map_axis_name)

        nsamples = self.nwalk * self.nav

        jac = self.vmap_getder(params, r, sz)

        f_i = jnp.real(-2 * psum(jnp.matmul(energy, jnp.conjugate(jac)), axis_name=self.map_axis_name) / nsamples)
        g2_i = self.beta * g2_i + (1. - self.beta) * f_i**2
        g2h_i = g2_i / ( 1. - self.beta**itr )

        S_ij = jnp.real(psum( jnp.matmul(jnp.transpose(jnp.conjugate(jac)), jac), axis_name=self.map_axis_name) / nsamples) 

        mean_diag = jnp.trace(S_ij) / S_ij.shape[0]
        jax.debug.print("mean_diag: {}", mean_diag)
        tiny = jnp.asarray(jnp.finfo(S_ij.dtype).eps, S_ij.dtype)
        lam = self.eps * (mean_diag + tiny)
        idx = jnp.diag_indices(S_ij.shape[0])
        S_ij = S_ij.at[idx].add(lam)
        f_i = f_i + (lam * self.alpha_spring) * m_i
        
        #S_ij += self.eps * jnp.diag( 1. + jnp.sqrt(g2h_i) )


        U_ij, low = cho_factor(S_ij)
        dp_i = cho_solve((U_ij, low), f_i)
        m_i = dp_i
        return dp_i, g2_i, m_i

    def sr_cholesky_variance(self, params, r, sz, energy, dp0_i, g2_i, m_i, itr):
        """Parameters' update according to the SR algorithm with Cholesky solver
        Args:
        params: initial variational parameters
        r: array with shape (nwalk/n_devices * nav, npart, ndim)
        sz: array with shape (nwalk/n_devices * nav, npart, 2)
        energy: array with shape (nwalk/n_devices * nav)
        g2_i: accumulated second order derivative squared  (nparams)
        itr : iteration

        Returns:
        dp_i: array with the same shape as ``params`` representing the best parameters' update (nparams)
        g2_i: updated accumulated second order derivative squared (nparams)
        """

        nsamples = self.nwalk * self.nav

        energy_mean = pmean(jnp.mean(energy), axis_name=self.map_axis_name)
        energy2_mean = pmean(jnp.mean(jnp.abs(energy)**2), axis_name=self.map_axis_name)
        test_variance = jnp.real( energy2_mean - energy_mean**2 )
        jax.debug.print("test_variance: {}", test_variance)
        
        jac_elocal = self.vmap_getder_elocal(params, r, sz)
        jac = self.vmap_getder(params, r, sz)

# Cyrus
#        jac_elocal = jac_elocal - pmean(jnp.mean(jac_elocal, axis=0), axis_name=self.map_axis_name)

# Ale changing omega
        energy_mean = -31.28
 
        energy = energy - energy_mean 
        f_i = jnp.matmul(jnp.conjugate(energy), jac_elocal) + jnp.matmul(jnp.abs(energy)**2, jac)  
        f_i = - 2 * jnp.real( psum(f_i, axis_name=self.map_axis_name) / nsamples )
        jac = jac_elocal + energy[:, None] * jac 

        g2_i = self.beta * g2_i + (1. - self.beta) * f_i**2
        g2h_i = g2_i / ( 1. - self.beta**itr )
        S_ij = jnp.real(psum( jnp.matmul(jnp.transpose(jnp.conjugate(jac)), jac), axis_name=self.map_axis_name) / nsamples) 
        S_ij += self.eps * jnp.diag( 1 + jnp.sqrt(g2h_i) )

        U_ij, low = cho_factor(S_ij)
        dp_i = cho_solve((U_ij, low), f_i)
        return dp_i, g2_i, m_i

    def sr_pseudo(self, params, r, sz, energy, dp0_i, g2_i, m_i, itr):
        """Parameters' update according to the SR algorithm with Pseudo-Inverse
        Args:
        params: initial variational parameters
        r: array with shape (nwalk/n_devices * nav, npart, ndim)
        sz: array with shape (nwalk/n_devices * nav, npart, 2)
        energy: array with shape (nwalk/n_devices * nav)

        Returns:
        dp_i: array with the same shape as ``params`` representing the best parameters' update (nparams)
        """

        energy = energy - jnp.mean(energy)
        nsamples = self.nwalk * self.nav
        energy = jnp.reshape(energy, (nsamples))

        jac = self.pmap_getder(params, r, sz, nsamples)
        jac_T = jnp.conjugate(jnp.transpose(jac))
        F = jnp.matmul(jac, jac_T)
        shift = nsamples * jnp.identity(nsamples)
        dp_i = jnp.linalg.solve(F + self.eps * shift, energy)

        dp_i = jnp.real(- 2 * jnp.matmul(jac_T, dp_i)) - self.lbd * self.wavefunction.flatten_params(params)
        return dp_i, g2_i, m_i

    def adam(self, params, r, sz, energy, dp0_i, g2_i, m_i, itr):
        """Parameters' update according to the Adam algorithm
        Args:
        params: initial variational parameters
        r: array with shape (nwalk/n_devices * nav, npart, ndim)
        sz: array with shape (nwalk/n_devices * nav, npart, 2)
        energy: array with shape (nwalk/n_devices * nav)
        m_i: accumulated momentum  (nparams)
        g2_i: accumulated second order derivative squared  (nparams)
        itr : iteration

        Returns:
        dp_i: array with the same shape as ``params`` representing the best parameters' update (nparams)
        g2_i: updated accumulated second order derivative squared (nparams)
        m_i: updated accumulated momentum (nparams)
        """

        energy = energy - pmean(jnp.mean(energy), axis_name=self.map_axis_name)

        nsamples = self.nwalk * self.nav

        jac = self.vmap_getder(params, r, sz)

        f_i = jnp.real(-2 * psum(jnp.matmul(energy, jnp.conjugate(jac)), axis_name=self.map_axis_name) / nsamples)

        m_i = self.alpha * m_i + (1. - self.alpha) * f_i
        mh_i = m_i / ( 1. - self.alpha**itr )
        g2_i = self.beta * g2_i + (1. - self.beta) * f_i**2
        g2h_i = g2_i / ( 1. - self.beta**itr )
        dp_i = mh_i / ( jnp.sqrt(g2h_i) + 0.00001 ) - self.lbd * self.wavefunction.flatten_params(params)
        return dp_i, g2_i, m_i

    def adam_variance(self, params, r, sz, energy, dp0_i, g2_i, m_i, itr):
        """Parameters' update according to the Adam algorithm
        Args:
        params: initial variational parameters
        r: array with shape (nwalk/n_devices * nav, npart, ndim)
        sz: array with shape (nwalk/n_devices * nav, npart, 2)
        energy: array with shape (nwalk/n_devices * nav)
        m_i: accumulated momentum  (nparams)
        g2_i: accumulated second order derivative squared  (nparams)
        itr : iteration

        Returns:
        dp_i: array with the same shape as ``params`` representing the best parameters' update (nparams)
        g2_i: updated accumulated second order derivative squared (nparams)
        m_i: updated accumulated momentum (nparams)
        """

        nsamples = self.nwalk * self.nav

        energy_mean = pmean(jnp.mean(energy), axis_name=self.map_axis_name)
        energy2_mean = pmean(jnp.mean(jnp.abs(energy)**2), axis_name=self.map_axis_name)
        test_variance = jnp.real( energy2_mean - energy_mean**2 )
        jax.debug.print("test_variance: {}", test_variance)
        
        jac_elocal = self.vmap_getder_elocal(params, r, sz)
        jac = self.vmap_getder(params, r, sz)
        jac = jac - pmean(jnp.mean(jac, axis=0), axis_name=self.map_axis_name) 

        energy_mean = -31.28
 
        energy = energy - energy_mean 
        f_i = jnp.matmul(jnp.conjugate(energy), jac_elocal) + jnp.matmul(jnp.abs(energy)**2, jac)  
        f_i = - 2 * jnp.real( psum(f_i, axis_name=self.map_axis_name) / nsamples )
 
        m_i = self.alpha * m_i + (1. - self.alpha) * f_i
        mh_i = m_i / ( 1. - self.alpha**itr )
        g2_i = self.beta * g2_i + (1. - self.beta) * f_i**2
        g2h_i = g2_i / ( 1. - self.beta**itr )
        dp_i = mh_i / ( jnp.sqrt(g2h_i) + 0.00001 ) - self.lbd * self.wavefunction.flatten_params(params)
        return dp_i, g2_i, m_i

    # Solve the SR equations
    def optimize(self, params, r_s, sz_s, energy_s, state):
        itr, dp0_i, g2_i, m_i = state

        r_av = jnp.reshape(r_s[:, 0:self.nav, :, :], (r_s.shape[0] * self.nav, self.npart, self.ndim))
        sz_av = jnp.reshape(sz_s[:, 0:self.nav, :, :], (sz_s.shape[0] * self.nav, self.npart, 2))
        energy_av = jnp.reshape(energy_s[:, 0:self.nav], (energy_s.shape[0] * self.nav,))

        # Validation and testing
        r_val = jnp.reshape(r_s[:, 0:self.nac, :, :], (r_s.shape[0] * self.nac, self.npart, self.ndim))
        sz_val = jnp.reshape(sz_s[:, 0:self.nac, :, :], (sz_s.shape[0] * self.nac, self.npart, 2))
        energy_val = jnp.reshape(energy_s[:, 0:self.nac], (energy_s.shape[0] * self.nac))
        logpsi_val, max_logpsi_val, avg_logpsi_val, min_logpsi_val = self.wavefunction.logpsi_stats(params, r_val, sz_val)
        validation_results = (max_logpsi_val, avg_logpsi_val, min_logpsi_val)

        r_tst = jnp.reshape(r_s[:, self.nav:self.nav+self.nac, :, :], (r_s.shape[0] * self.nac, self.npart, self.ndim))
        sz_tst = jnp.reshape(sz_s[:, self.nav:self.nav+self.nac, :, :], (sz_s.shape[0] * self.nac, self.npart, 2))
        energy_tst = jnp.reshape(energy_s[:, self.nav:self.nav+self.nac], (energy_s.shape[0] * self.nac))
        logpsi_tst, max_logpsi_tst, avg_logpsi_tst, min_logpsi_tst = self.wavefunction.logpsi_stats(params, r_tst, sz_tst)
        test_results = (max_logpsi_tst, avg_logpsi_tst, min_logpsi_tst)

        # Wavefunction parameter optimization
        dp_i, g2_i, m_i = self.sr_solver(params, r_av, sz_av, energy_av, dp0_i, g2_i, m_i, itr)

        delta_p = self.delta * dp_i
        dp_max = jnp.max(jnp.abs(delta_p))
        dp_norm = jnp.linalg.norm(delta_p)
        delta_p = self.wavefunction.unflatten_params(delta_p)

        # Optimization result testing
        dist_tst_result = self.dist(delta_p, params, r_tst, sz_tst, logpsi_tst, energy_tst)
        dist_val_result = self.dist(delta_p, params, r_val, sz_val, logpsi_val, energy_val)

        optimization_ouput = (validation_results, test_results,
                              dist_val_result, dist_tst_result,
                              dp_max, dp_norm)

        return (optimization_ouput, (dp_i, g2_i, m_i), delta_p)

    def shmap_optimize(self, params, r_s, sz_s, energy_s):
        """
        Optimizes the wave function according to different training
        algorithms

        Args:
        params: initial variational parameters
        r_s: array with shape (nwalk, nav + nac, npart, ndim)
        sz_s: array with shape (nwalk, nav + nac, npart, 2)
        energy_s: array with shape (nwalk, nav + nac)

        Returns:
        Array with the same shape as ``params`` representing the best parameters' update
        """

        self.itr += 1
        optimizer_state = (self.itr, self.dp_i, self.g2_i, self.m_i)

        optimization_ouput, state_update, delta_p = self.shard_map_optimize(params, r_s, sz_s, energy_s, optimizer_state)

        # Save updated optimizer initialization
        self.dp_i = state_update[0]
        self.g2_i = state_update[1]
        self.m_i = state_update[2]

        # Unpacking output
        (validation_results, test_results,
            dist_val_result, dist_tst_result,
            dp_max, dp_norm) = optimization_ouput
        max_logpsi_val, avg_logpsi_val, min_logpsi_val = validation_results
        max_logpsi_tst, avg_logpsi_tst, min_logpsi_tst = test_results
        energy_d_tst, energy_d_err_tst, fidelity_dp_tst, fidelity_dt_tst = dist_val_result
        energy_d_val, energy_d_err_val, fidelity_dp_val, fidelity_dt_val = dist_tst_result

        # Log optimization output
        logger.info(f"Maximum |Psi| validation = {jnp.exp(max_logpsi_val):.4e}")
        logger.info(f"Average |Psi| validation = {jnp.exp(avg_logpsi_val):.4e}")
        logger.info(f"Minimum |Psi| validation = {jnp.exp(min_logpsi_val):.4e}")

        logger.info(f"Maximum |Psi| test = {jnp.exp(max_logpsi_tst):.4e}")
        logger.info(f"Average |Psi| test = {jnp.exp(avg_logpsi_tst):.4e}")
        logger.info(f"Minimum |Psi| test = {jnp.exp(min_logpsi_tst):.4e}")

        logger.debug(f"energy diff test = {energy_d_tst:.6f}, err = {energy_d_err_tst:.6f}")
        logger.debug(f"fidelity[psi(p), psi(p+dp)] test = {fidelity_dp_tst:.6f}")
        logger.debug(f"fidelity[(1 - H dt)psi(p), psi(p+dp)] test = {fidelity_dt_tst:.6f}")

        logger.debug(f"energy diff validation = {energy_d_val:.6f}, err = {energy_d_err_val:.6f}")
        logger.debug(f"F[psi(p), psi(p+dp)] validation = {fidelity_dp_val:.6f}")
        logger.debug(f"F[(1 - H dt)psi(p), val(p+dp)] validation = {fidelity_dt_val:.6f}")

        logger.debug(f"delta param max = {dp_max:.6f}")
        logger.debug(f"delta param norm = {dp_norm:.6f}")

        # Test for convergence and return parameter update
        if fidelity_dp_val > 0.9 and fidelity_dp_tst > 0.9 and dp_max < 0.5:
            logger.debug(f"Converged, energy diff min = {energy_d_val:.6f}, err = {energy_d_err_val:.6f}")
            return delta_p

        logger.debug(f"Not converged")
        return self.wavefunction.unflatten_params(jnp.zeros(self.nparams))

    @partial(jax.jit, static_argnums=(0,))
    def dist(self, delta_p, params, r, sz, logpsi_o, energy_o):
        nsamples = self.nwalk * self.nac

        # Compute the old wave function and energy
        energy_o_sum = psum(jnp.sum(energy_o), axis_name=self.map_axis_name) / nsamples
        energy2_o_sum = psum(jnp.sum(energy_o**2), axis_name=self.map_axis_name) / nsamples
        energy_o_err = jnp.sqrt((energy2_o_sum - energy_o_sum**2) / nsamples)

        # Update the parameters
        params_n = jax.tree_util.tree_map(self.wavefunction.update_add, params, delta_p)

        # Compute the new wave function and energy reweighting the stored walk
        logpsi_n = self.wavefunction.vmap_logpsi(params_n, r, sz)
        energy_n = self.observables.energy(params_n, r, sz)[:, 0]
        psi_ratio = jnp.exp(logpsi_n - logpsi_o)
        psi2_norm_sum = psum(jnp.sum(jnp.abs(psi_ratio)**2), axis_name=self.map_axis_name) / nsamples
        psi_norm_sum = psum(jnp.sum(psi_ratio), axis_name=self.map_axis_name) / nsamples

#        energy_n *= jnp.abs(psi_ratio)**2
        energy_n_sum = psum( jnp.sum(energy_n * jnp.abs(psi_ratio)**2), axis_name=self.map_axis_name) / nsamples / psi2_norm_sum
        energy2_n_sum = psum( jnp.sum(energy_n**2 * jnp.abs(psi_ratio)**2), axis_name=self.map_axis_name) / nsamples / psi2_norm_sum
        energy_n_err = jnp.sqrt((energy2_n_sum - energy_n_sum**2) / nsamples )

        # Correlated energy difference
        energy_d = energy_n * jnp.abs(psi_ratio)**2 / psi2_norm_sum - energy_o
        energy_d_sum = psum(jnp.sum(energy_d), axis_name=self.map_axis_name) / nsamples
        energy2_d_sum = psum(jnp.sum(energy_d**2), axis_name=self.map_axis_name) / nsamples
        energy_d_err = jnp.sqrt((energy2_d_sum - energy_d_sum**2) / nsamples )

        # Fidelity between |Psi_{p+dp}> and |Psi_p> using samples from  \Pi_{Psi_p}(X)
        fidelity_dp = jnp.abs(psi_norm_sum)**2 / psi2_norm_sum

        # Fidelity between (1 - H dt)|Psi_p> and |Psi_{p+dp}> using samples from \Pi_{Psi_p}(X)
        psi_dt = (1. - energy_o * self.delta)
        overlap_dt = psum(jnp.sum(jnp.conjugate(psi_ratio) * psi_dt), axis_name=self.map_axis_name) / nsamples
        norm_n = psum(jnp.sum(jnp.abs(psi_ratio)**2), axis_name=self.map_axis_name) / nsamples
        norm_dt = psum(jnp.sum(jnp.abs(psi_dt)**2), axis_name=self.map_axis_name) / nsamples
        fidelity_dt = jnp.abs(overlap_dt)**2 / norm_n / norm_dt

        return energy_d_sum, energy_d_err, fidelity_dp, fidelity_dt
