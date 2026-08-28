import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from jax.lax import fori_loop, psum, pmean, pmax
from jax import random, grad, jit, vmap, jacfwd, jacrev, pmap
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from jax.lax import fori_loop, cond
from functools import partial
from itertools import product
from folx import forward_laplacian

import logging
logger = logging.getLogger()

class Observables(object):
    """Observables class
    """

    def __init__(self, n_devices, wavefunction, potential, config, mesh):
        self.ndim = config.ndim
        if self.ndim < 1 or self.ndim > 3:
            raise Exception("Dimension must be 1, 2, or 3")

        self.n_devices = n_devices
        self.mass_p = config.mass_p
        self.mass_n = config.mass_n
        reduced_mass = self.mass_p * self.mass_n / (self.mass_p + self.mass_n)
        self.hbar = config.hbar
        self.hbar2m = 0.25 * self.hbar**2 / reduced_mass
        self.omega = config.omega
        self.nwalk = config.nwalk
        self.npart = config.npart
        self.nrho = config.nrho
        self.conserved_sz = config.conserved_sz
        self.wavefunction = wavefunction
        self.potential = potential

        self.npair = int(self.npart * (self.npart - 1) / 2)
        # Number which divides the number of pairs weather npart is even or odd
        self.divisor = (self.npart-1)*((self.npart+1) % 2) + (self.npart)*(self.npart % 2)

        self.ip = jnp.empty(self.npair, dtype=int)
        self.jp = jnp.empty(self.npair, dtype=int)
        k = 0
        for i in range(self.npart-1):
            for j in range(i+1, self.npart):
                self.ip = self.ip.at[k].set(i)
                self.jp = self.jp.at[k].set(j)
                k += 1

        self.ntriplet = (self.npart*(self.npart-1)*(self.npart-2)) // 6
        self.triplet_inds = jnp.empty((self.ntriplet, 3), dtype=int)
        nt = 0
        for i in range(self.npart-2):
            for j in range(i+1, self.npart-1):
                for k in range(j+1, self.npart):
                    self.triplet_inds = self.triplet_inds.at[nt, :].set([i, j, k])
                    nt += 1

        if self.conserved_sz:
            self.v_pair = self.v_pair_conserved_sz
            logger.info('Conserved Sz in the NN potential')
        else:
            self.v_pair = self.v_pair_tensor
            logger.info('Allow for tensor force in the NN potential')

        # Selecting three body version
        if potential.pot_3b_type == 'Linear':
            self.three_body = self.linear_three_body
        elif potential.pot_3b_type == 'Linear_with_projector':
            self.three_body = self.linear_proj_three_body
        elif potential.pot_3b_type == 'Triangle_with_projector':
            self.three_body = self.triangle_proj_three_body

        self.remove_cm = config.remove_cm
        self.periodic = config.periodic
        self.L = config.L
        if self.periodic:
            self.box_dist = config.box_dist
            nums = list(range(-1*self.box_dist, self.box_dist+1))
            self.box_vecs = self.L*np.array(list(product(nums, repeat=self.ndim)))

            # Limits included contributions to a radius in which all
            # contributions are able to be included.
            self.dist_threshold = (self.box_dist+0.5)*self.L

            self.v_ii = 0
            for vec in self.box_vecs:
                dist = jnp.sqrt(np.sum(vec**2))
                # The dist > self.L/2 only excludes a distance of zero since
                # all other distances are greater than or equal to self.L
                if self.dist_threshold >= dist and dist > self.L/2:
                    self.v_ii += self.potential.v_2b(dist)

            # Include one of these factors for each particle
            # Divide by 2 because it is shared with the image particle
            self.v_ii *= 1/2

        # Set the Laplacian computation
        if config.laplacian == 'Jax':
            self.kinetic_energy = self.kinetic_energy_jax
        elif config.laplacian == 'Folx':
            self.kinetic_energy = self.kinetic_energy_folx
        elif config.laplacian == 'Gabriel':
            self.kinetic_energy = self.kinetic_energy_gabriel
        elif config.laplacian == 'Stencil':
            self.kinetic_energy = self.kinetic_energy_stencil
        else:
            logger.error('Invalid Laplacian method. Please choose either "jax" or "folx".')
            raise ValueError('Invalid Laplacian method. Please choose either "jax" or "folx".')

        self.L2_coupling = config.L2_coupling
        self.Lz_coupling = config.Lz_coupling

        self.levi = jnp.array([[1, 2, 0], [2, 0, 1]])

        self.basis = jnp.zeros(shape=[self.npart * self.ndim, self.npart * self.ndim])
        for i in range(self.npart * self.ndim):
            self.basis = self.basis.at[i,i].set(1)
        self.basis = jnp.reshape(self.basis, (self.ndim * self.npart, self.npart, self.ndim))

        # Several objects get stored for referencing, if needed, after energy computation:
        self.rho_min = 0.
        if self.periodic:
            self.rho_max = self.dist_threshold
        else:
            self.rho_max = 10.
        self.r_shells = np.linspace(self.rho_min, self.rho_max, num=self.nrho+1)
        self.v_rho = np.zeros(shape=[self.nrho])
        for i in range(self.nrho):
            self.v_rho[i] = self.r_shells[i+1]**3 - self.r_shells[i]**3
        self.v_rho = 4. / 3. * jnp.pi * self.v_rho
        # set r_rho to centers of shells
        self.r_rho = (self.r_shells[:-1]+self.r_shells[1:])/2


        self.angular_momentum = jax.jit(self.angular_momentum)
        self.angular_momentum_vmap = jax.jit(self.angular_momentum_vmap)
        self.two_body_density_vmap_singlet_triplet = jax.jit(self.two_body_density_vmap_singlet_triplet)
        self.two_body_density_vmap_isospin = jax.jit(self.two_body_density_vmap_isospin)

        # prep shardmap
        self.map_axis_name = mesh.axis_names[0]
        Pspec = P(self.map_axis_name)
        Pnone = P()

        self.energy_shmap = shard_map(self.energy_compute, mesh=mesh,
                                      in_specs=(Pnone, Pspec, Pspec),
                                      out_specs=(Pspec, Pnone, Pnone),
                                      check_rep=False)
        self.energy_shmap = jax.jit(self.energy_shmap)

        self.density_shmap = shard_map(self.density_compute, mesh=mesh,
                                       in_specs=(Pspec, Pspec),
                                       out_specs=(Pnone, Pnone, Pnone))
        self.density_shmap = jax.jit(self.density_shmap)

        self.radius_shmap = shard_map(self.radius_compute, mesh=mesh,
                                      in_specs=(Pspec, Pspec),
                                      out_specs=(Pnone, Pnone))
        self.radius_shmap = jax.jit(self.radius_shmap)

        self.variance_shmap = shard_map(self.variance_compute, mesh=mesh,
                                        in_specs=(Pspec),
                                        out_specs=(Pnone))
        self.variance_shmap = jax.jit(self.variance_shmap)

        self.two_body_isospin_shmap = shard_map(self.two_body_density_isospin_compute, mesh=mesh,
                                                in_specs=(Pspec, Pspec),
                                                out_specs=(Pnone, Pnone))
        self.two_body_isospin_shmap = jax.jit(self.two_body_isospin_shmap)

    # Spin or isospin exchange operation
    @partial(jit, static_argnums=(0,))
    def sp_exch(self, sz_o, k, ist):
        sz_n = sz_o.at[self.ip[k], ist].set(sz_o[self.jp[k], ist])
        sz_n = sz_n.at[self.jp[k], ist].set(sz_o[self.ip[k], ist])
        return sz_n

    @partial(jax.jit, static_argnums=(0,))
    def v_pair_conserved_sz(self, k, params, r, sz, logpsi):
        r_ij = r[self.ip[k], :] - r[self.jp[k], :]
        if not self.periodic:
            r_ij = jnp.sqrt(jnp.sum(r_ij**2))
            vr_ij = self.potential.v_2b(r_ij)
            vrem_ij_pp, vrem_ij_np, vrem_ij_nn = self.potential.v_em(r_ij)

            tz_i = sz[self.ip[k], 1]
            tz_j = sz[self.jp[k], 1]
            proton_i = (1+tz_i)/2
            neutron_i = (1-tz_i)/2
            proton_j = (1+tz_j)/2
            neutron_j = (1-tz_j)/2
            nn_proj = neutron_i*neutron_j
            np_proj = (neutron_i*proton_j+proton_i*neutron_j)
            pp_proj = proton_i*proton_j

            vem_ij = vrem_ij_pp*pp_proj + vrem_ij_np*np_proj + vrem_ij_nn*nn_proj

            # Add EM terms to the potential terms with the same spin/isospin
            # operator dependence
            vr_ij += vem_ij

            # Three body
            t_ij = self.potential.v_3b(r_ij)
            t2_ij = t_ij**2
        else:
            r_ij = r_ij - jnp.rint(r_ij/self.L)*self.L
            r_ij = r_ij + self.box_vecs
            r_ij = jnp.sqrt(jnp.sum(r_ij**2, axis=1))

            trunc_mult = vmap(lambda r: cond(r > self.dist_threshold, lambda: 0, lambda: 1))(r_ij)
            trunc_mult = trunc_mult.reshape((-1, 1))

            # Currently em component is always zero for periodic evaluation
            # vem_ij = jnp.zeros(6)
            vr_ij = sum(trunc_mult*vmap(self.potential.v_2b)(r_ij))

            temp_t_ij = trunc_mult*(vmap(self.potential.v_3b)(r_ij).reshape((-1, 1)))
            t_ij = sum(temp_t_ij)
            t2_ij = sum(temp_t_ij**2)

        sz_ij = self.sp_exch(sz, k, 0)
        tz_ij = self.sp_exch(sz, k, 1)
        stz_ij = self.sp_exch(tz_ij, k, 0)

        logpsi_t_ij = self.wavefunction.logpsi(params, r, tz_ij)
        Pt_ij = jnp.exp(logpsi_t_ij - logpsi)

        logpsi_s_ij = self.wavefunction.logpsi(params, r, sz_ij)
        Ps_ij = jnp.exp(logpsi_s_ij - logpsi)

        logpsi_st_ij = self.wavefunction.logpsi(params, r, stz_ij)
        Pst_ij = jnp.exp(logpsi_st_ij - logpsi)

        vc_ij = vr_ij[0]
        vt_ij = vr_ij[1] * (2 * Pt_ij - 1)
        vs_ij = vr_ij[2] * (2 * Ps_ij - 1)
        vst_ij = vr_ij[3] * (4 * Pst_ij - 2 * Pt_ij - 2 * Ps_ij + 1)
        vT_ij = vr_ij[6] * (3*sz[self.ip[k], 1]*sz[self.jp[k], 1]-(2*Pt_ij-1))
        vtz_ij = vr_ij[7] * (sz[self.ip[k], 1] + sz[self.jp[k], 1])

        return vc_ij, vt_ij, vs_ij, vst_ij, vT_ij, vtz_ij, t_ij, t2_ij

    @partial(jax.jit, static_argnums=(0,))
    def v_pair_tensor(self, k, params, r, sz, logpsi):
        r_vec_ij = r[self.ip[k], :] - r[self.jp[k], :]
        if not self.periodic:
            r_ij = jnp.sqrt(jnp.sum(r_vec_ij**2))
            vr_ij = self.potential.v_2b(r_ij)
            vrem_ij_pp, vrem_ij_np, vrem_ij_nn = self.potential.v_em(r_ij)

            tz_i = sz[self.ip[k], 1]
            tz_j = sz[self.jp[k], 1]
            proton_i = (1+tz_i)/2
            neutron_i = (1-tz_i)/2
            proton_j = (1+tz_j)/2
            neutron_j = (1-tz_j)/2
            nn_proj = neutron_i*neutron_j
            np_proj = (neutron_i*proton_j+proton_i*neutron_j)
            pp_proj = proton_i*proton_j

            vem_ij = vrem_ij_pp*pp_proj + vrem_ij_np*np_proj + vrem_ij_nn*nn_proj

            # Add EM terms to the potential terms with the same spin/isospin
            # operator dependence
            vr_ij += vem_ij

            # Three body
            t_ij = self.potential.v_3b(r_ij)
            t2_ij = t_ij**2
        else:
            r_ij = r_vec_ij - jnp.rint(r_vec_ij/self.L)*self.L
            r_ij = r_vec_ij + self.box_vecs
            r_ij = jnp.sqrt(jnp.sum(r_vec_ij**2, axis=1))

            trunc_mult = vmap(lambda r: cond(r > self.dist_threshold, lambda: 0, lambda: 1))(r_ij)
            trunc_mult = trunc_mult.reshape((-1, 1))

            # Currently em component is always zero for periodic evaluation
            # vem_ij = jnp.zeros(6)
            vr_ij = sum(trunc_mult*vmap(self.potential.v_2b)(r_ij))

            temp_t_ij = trunc_mult*(vmap(self.potential.v_3b)(r_ij).reshape((-1, 1)))
            t_ij = sum(temp_t_ij)
            t2_ij = sum(temp_t_ij**2)

        vc_ij = vr_ij[0]
        sz_sigma_i, coef_sigma_i = self.opmult(sz[self.ip[k], 0])
        sz_sigma_j, coef_sigma_j = self.opmult(sz[self.jp[k], 0])

        sz_Pt_ij = self.sp_exch(sz, k, 1)
        logpsi_t_ij = self.wavefunction.logpsi(params, r, sz_Pt_ij)
        Pt_ij = jnp.exp(logpsi_t_ij - logpsi)
        tau_ij = (2 * Pt_ij - 1)

        #jax.debug.print("tau_ij: {}", tau_ij)

        sz_sigma_tau_i, coef_sigma_tau_i = self.opmult(sz_Pt_ij[self.ip[k], 0])
        sz_sigma_tau_j, coef_sigma_tau_j = self.opmult(sz_Pt_ij[self.jp[k], 0])

        sigma_ij = 0j
        sigma_tau_ij = 0j
        S_ij = 0j
        S_tau_ij = 0j
        for i in range(3):
            sz_sigma_ij = sz.at[self.ip[k],0].set(sz_sigma_i[i])
            sz_sigma_ij = sz_sigma_ij.at[self.jp[k],0].set(sz_sigma_j[i])
            logpsi_sigma_ij = self.wavefunction.logpsi(params, r, sz_sigma_ij)
            sigma_ij = sigma_ij + coef_sigma_i[i] * coef_sigma_j[i] * jnp.exp(logpsi_sigma_ij - logpsi)

            sz_sigma_tau_ij = sz_Pt_ij.at[self.ip[k],0].set(sz_sigma_tau_i[i])
            sz_sigma_tau_ij = sz_sigma_tau_ij.at[self.jp[k],0].set(sz_sigma_tau_j[i])
            logpsi_sigma_tau_ij = self.wavefunction.logpsi(params, r, sz_sigma_tau_ij)
            sigma_tau_ij = sigma_tau_ij + coef_sigma_tau_i[i] * coef_sigma_tau_j[i] * jnp.exp(logpsi_sigma_tau_ij - logpsi)
            for j in range(3):
                sz_sigma_ij = sz_sigma_ij.at[self.jp[k],0].set(sz_sigma_j[j])
                logpsi_sigma_ij = self.wavefunction.logpsi(params, r, sz_sigma_ij)
                S_ij = S_ij + r_vec_ij[i] *  r_vec_ij[j] / r_ij**2  * coef_sigma_i[i] * coef_sigma_j[j] * jnp.exp(logpsi_sigma_ij - logpsi)

                sz_sigma_tau_ij = sz_sigma_tau_ij.at[self.jp[k],0].set(sz_sigma_tau_j[j])
                logpsi_sigma_tau_ij = self.wavefunction.logpsi(params, r, sz_sigma_tau_ij)
                S_tau_ij = S_tau_ij + r_vec_ij[i] *  r_vec_ij[j] / r_ij**2  * coef_sigma_tau_i[i] * coef_sigma_tau_j[j] * jnp.exp(logpsi_sigma_tau_ij - logpsi)

        S_ij = 3 * S_ij - sigma_ij
        S_tau_ij = 3 * S_tau_ij - sigma_tau_ij
        sigma_tau_ij = 2 * sigma_tau_ij - sigma_ij
        S_tau_ij = 2 * S_tau_ij - S_ij

        vt_ij = vr_ij[1] * tau_ij
        vs_ij = vr_ij[2] * sigma_ij + vr_ij[4] * S_ij
        vst_ij = vr_ij[3] * sigma_tau_ij + + vr_ij[5] * S_tau_ij
        v_S_ij = vr_ij[4] * S_ij
        v_St_ij = vr_ij[5] * S_tau_ij

        vT_ij = vr_ij[6] * (3*sz[self.ip[k], 1]*sz[self.jp[k], 1]-(2*Pt_ij-1))
        vtz_ij = vr_ij[7] * (sz[self.ip[k], 1] + sz[self.jp[k], 1])

        return vc_ij, vt_ij, vs_ij, vst_ij, vT_ij, vtz_ij, t_ij, t2_ij

    def opmult(self, sz):
        """Computes <S| sigma_alpha, remember that here the spin is in the bra
        hence the complex conjugate"""
        ci = 1j
        szmult = jnp.ones((3, 2))
        cmult = jnp.zeros((3, 2), dtype=jnp.complex64)

        sz = ((sz + 1) / 2 ).astype(int)

        szmult = szmult.at[0, 0].set(1)  # <down|sigma_x = |up>
        cmult = cmult.at[0, 0].set(1)

        szmult = szmult.at[0, 1].set(-1)  # <up|sigma_x = |down>
        cmult = cmult.at[0, 1].set(1)

        szmult = szmult.at[1, 0].set(1)  # <down|sigma_y = i |up>
        cmult = cmult.at[1, 0].set(1j)

        szmult = szmult.at[1, 1].set(-1)  # <up|sigma_y = -i |down>
        cmult = cmult.at[1, 1].set(-1j)

        szmult = szmult.at[2, 0].set(-1)  # <down| sigma_z = - |down>
        cmult = cmult.at[2, 0].set(-1)

        szmult = szmult.at[2, 1].set(1)  # <up| sigma_z = |up>
        cmult = cmult.at[2, 1].set(1)

        return szmult[:, sz], cmult[:, sz]

    @partial(jax.jit, static_argnums=(0,))
    def linear_three_body(self, t_ij, t2_ij, sz):
        gr3b = jnp.zeros(self.npart, dtype=jnp.complex128)
        V_ijk = 0
        if (self.npart > 2):
            for k in range(self.npair):
                gr3b = gr3b.at[self.ip[k]].add(t_ij[k])
                gr3b = gr3b.at[self.jp[k]].add(t_ij[k])
            V_ijk = 0.5 * jnp.sum(gr3b**2) - jnp.sum(t2_ij)
        return V_ijk

    @partial(jax.jit, static_argnums=(0,))
    def linear_proj_three_body(self, t_ij_temp, t2_ij, sz):
        # reshape t_ij into an npart x npart matrix
        t_ij_temp = t_ij_temp.reshape(self.npair)
        t_ij = jnp.zeros((self.npart, self.npart))
        t_ij = t_ij.at[jnp.triu_indices(self.npart, k=1)].set(t_ij_temp)
        t_ij = t_ij + t_ij.T

        V_ijk = 0.0
        if (self.npart > 2):
            def body_fun(nt, V_ijk):
                # Get particle indices for this triplet
                i, j, k = self.triplet_inds[nt]
                # Projector which is zero for nnn and ppp as well as three identical spins but one otherwise
                projector_array = (3-sz[i]*sz[j]-sz[i]*sz[k]-sz[j]*sz[k])//4
                projector = projector_array[0]*projector_array[1]

                # Potential contribution (only two body)
                V_ijk += projector*(t_ij[i, j]*t_ij[j, k] + t_ij[i, k]*t_ij[j, k] + t_ij[i, j]*t_ij[i, k])
                return V_ijk

            V_ijk = fori_loop(0, len(self.triplet_inds), body_fun, V_ijk)

        return V_ijk

    @partial(jax.jit, static_argnums=(0,))
    def triangle_proj_three_body(self, t_ij_temp, t2_ij, sz):
        # reshape t_ij into an npart x npart matrix
        t_ij_temp = t_ij_temp.reshape(self.npair)
        t_ij = jnp.zeros((self.npart, self.npart))
        t_ij = t_ij.at[jnp.triu_indices(self.npart, k=1)].set(t_ij_temp)
        t_ij = t_ij + t_ij.T

        V_ijk = 0.0
        if (self.npart > 2):
            def body_fun(nt, V_ijk):
                # Get particle indices for this triplet
                i, j, k = self.triplet_inds[nt]
                # Projector which is zero for nnn and ppp as well as three identical spins but one otherwise
                projector_array = (3-sz[i]*sz[j]-sz[i]*sz[k]-sz[j]*sz[k])//4
                projector = projector_array[0]*projector_array[1]

                # Potential contribution (true 3 body (Triangle))
                V_ijk += projector*(t_ij[i, j]*t_ij[j, k]*t_ij[i, k])
                return V_ijk

            V_ijk = fori_loop(0, len(self.triplet_inds), body_fun, V_ijk)

        return V_ijk

    @partial(jax.jit, static_argnums=(0,))
    def potential_energy(self, params, r, sz):
        "Returns potential energy"

        v_ij = jnp.zeros(6, dtype=jnp.complex128)
        k = jnp.arange(self.npair)

        logpsi = self.wavefunction.logpsi( params, r, sz )
        v_pair_map = lambda k: self.v_pair(k, params, r, sz, logpsi)
        vc_ij, vt_ij, vs_ij, vst_ij, vT_ij, vtz_ij, t_ij, t2_ij = jax.lax.map(vmap(v_pair_map), k.reshape((-1, self.divisor)))
#        vc_ij, vt_ij, vs_ij, vst_ij, vT_ij, vtz_ij, t_ij, t2_ij = vmap(v_pair_map)(k)
#        vc_ij, vt_ij, vs_ij, vst_ij, vT_ij, vtz_ij, t_ij, t2_ij = jax.lax.map(v_pair_map, k)

        vc_ij = vc_ij.reshape(self.npair)
        vt_ij = vt_ij.reshape(self.npair)
        vs_ij = vs_ij.reshape(self.npair)
        vst_ij = vst_ij.reshape(self.npair)
        vT_ij = vT_ij.reshape(self.npair)
        vtz_ij = vtz_ij.reshape(self.npair)
        t_ij = t_ij.reshape(self.npair)
        t2_ij = t2_ij.reshape(self.npair)

        v_ij = v_ij.at[0].add(jnp.sum(vc_ij[:]))
        v_ij = v_ij.at[1].add(jnp.sum(vt_ij[:]))
        v_ij = v_ij.at[2].add(jnp.sum(vs_ij[:]))
        v_ij = v_ij.at[3].add(jnp.sum(vst_ij[:]))
        v_ij = v_ij.at[4].add(jnp.sum(vT_ij[:]))
        v_ij = v_ij.at[5].add(jnp.sum(vtz_ij[:]))

        V_ijk = self.three_body(t_ij, t2_ij, sz)

        if self.periodic and self.box_dist > 0:
            v_ij = v_ij.at[0].add(self.v_ii[0]*self.npart)
            v_ij = v_ij.at[1].add(self.v_ii[1]*self.npart)
            v_ij = v_ij.at[2].add(self.v_ii[2]*self.npart)
            v_ij = v_ij.at[3].add(self.v_ii[3]*self.npart)
            v_ij = v_ij.at[4].add(self.v_ii[4]*2*self.npart)
            v_ij = v_ij.at[5].add(self.v_ii[5]*2*jnp.sum(sz[:, 1]))

        pe = v_ij[0] + v_ij[1] + v_ij[2] + v_ij[3] + v_ij[4] + v_ij[5] + V_ijk

#        if (self.vext):
#           vext = self.omega * jnp.sum(r**2)
#           pe += vext

        return pe

    @partial(jax.jit, static_argnums=(0,))
    def kinetic_energy_folx(self, params, r, sz):
        "Returns kinetic energy using the forward laplacian method by folx"

        logpsi_r = lambda r: jnp.real(self.wavefunction.logpsi(params, r, sz))
        logpsi_i = lambda r: jnp.imag(self.wavefunction.logpsi(params, r, sz))

        forward_pass_r = forward_laplacian(logpsi_r)(r)
        dlogpsi_r = forward_pass_r.jacobian.dense_array
        d2logpsi_r = forward_pass_r.laplacian

        forward_pass_i = forward_laplacian(logpsi_i)(r)
        dlogpsi_i = forward_pass_i.jacobian.dense_array
        d2logpsi_i = forward_pass_i.laplacian

        dlogpsi = dlogpsi_r + 1j * dlogpsi_i
        d2logpsi = d2logpsi_r + 1j * d2logpsi_i

        ke = - self.hbar2m * ( d2logpsi + jnp.sum( dlogpsi * dlogpsi ) )
        ke_jf = self.hbar2m * jnp.sum( jnp.conj(dlogpsi) * dlogpsi )
        return ke, ke_jf

    @partial(jax.jit, static_argnums=(0,))
    def kinetic_energy_jax(self, params, r, sz):
        "Returns kinetic energy"

        logpsi_r = lambda r: jnp.real(self.wavefunction.logpsi(params, r, sz))
        logpsi_i = lambda r: jnp.imag(self.wavefunction.logpsi(params, r, sz))

        dlogpsi_r = jax.grad(logpsi_r)(r)
        d2logpsi_r = jax.hessian(logpsi_r)(r)
        d2logpsi_r = jnp.reshape(d2logpsi_r, (self.ndim * self.npart, self.ndim * self.npart))

        dlogpsi_i = jax.grad(logpsi_i)(r)
        d2logpsi_i = jax.hessian(logpsi_i)(r)
        d2logpsi_i = jnp.reshape(d2logpsi_i, (self.ndim * self.npart, self.ndim * self.npart))

        dlogpsi = dlogpsi_r + 1j * dlogpsi_i
        d2logpsi = d2logpsi_r + 1j * d2logpsi_i

        proton_proj = (1+sz[:, 1])/2
        neutron_proj = (1-sz[:, 1])/2
        hbar2m = self.hbar**2*((1/(2*self.mass_p))*proton_proj+(1/(2*self.mass_n))*neutron_proj)
        hbar2m_repeat = jnp.repeat(hbar2m, self.ndim)
        hbar2m_T = jnp.reshape(hbar2m, (-1, 1))

        ke = - (jnp.trace(hbar2m_repeat * d2logpsi) + jnp.sum(hbar2m_T * dlogpsi * dlogpsi))
        ke_jf = jnp.sum(hbar2m_T * jnp.conj(dlogpsi) * dlogpsi)

        return ke, ke_jf

    def jacrev(self, f):
        def jacfun(x):
            y, vjp_fun = jax.vjp(f, x)
            eye = jnp.eye(y.size)[0]
            J = jax.vmap(vjp_fun, in_axes=0)(eye)
            return J
        return jacfun

    def jacfwd(self, f):
        def jacfun(x):
            jvp_fun = lambda s: jax.jvp(f, (x,), (s,))[1]
            eye = jnp.eye(len(x))
            J = jax.lax.map(jvp_fun, eye)
            return J
        return jacfun

    def kinetic_energy_gabriel(self, params, x, sz):
        x = x.reshape((self.npart * self.ndim))

        def logpsi_r(x):
            x = x.reshape((self.npart, self.ndim))
            return jnp.real(self.wavefunction.logpsi(params, x, sz))

        def logpsi_i(x):
            x = x.reshape((self.npart, self.ndim))
            return jnp.imag(self.wavefunction.logpsi(params, x, sz))

        dlogpsi_r = self.jacrev(logpsi_r)
        d2logpsi_r = jnp.diag(self.jacfwd(dlogpsi_r)(x)[0].reshape(x.shape[0], x.shape[0]))

        dlogpsi_i = self.jacrev(logpsi_i)
        d2logpsi_i = jnp.diag(self.jacfwd(dlogpsi_i)(x)[0].reshape(x.shape[0], x.shape[0]))

        dlogpsi = dlogpsi_r(x)[0][0] + 1j * dlogpsi_i(x)[0][0]
        d2logpsi = d2logpsi_r + 1j * d2logpsi_i

        ke =  -self.hbar2m * jnp.sum(d2logpsi + dlogpsi * dlogpsi, axis=-1)
        ke_jf = self.hbar2m * jnp.sum(jnp.conj(dlogpsi) * dlogpsi,  axis=-1)

        return ke, ke_jf

    def hessian_diag(self, f, x):
        def hvp(f, x, v):
            return jax.jvp(jax.grad(f, holomorphic = True), [x], [v])[1]
        comp = lambda v: jnp.vdot(v, hvp(f, x, v))
        return jax.vmap(comp)(self.basis)

    def elocal(self, params, r, sz):
        ke, ke_jf = self.kinetic_energy(params, r, sz)
        pe = self.potential_energy(params, r, sz)
        energy_jf = ke_jf + pe
        energy_pt = ke + pe
        return energy_pt, energy_jf

    def vmap_elocal(self, params, r, sz):
        return vmap(self.elocal, in_axes=(None, 0, 0))(params, r, sz)

    def energy(self, params, r, sz):
        "Returns the total energy by vmapping over the input walkers"

        energy_pt, energy_jf = self.vmap_elocal(params, r, sz)

        L2 = jnp.zeros(energy_pt.shape[0], dtype=jnp.complex128)
        Lvec = jnp.zeros(shape=(energy_pt.shape[0], 3), dtype=jnp.complex128)
        L2, Lvec = self.angular_momentum_vmap(params, r, sz)

        energy = energy_pt + self.L2_coupling * L2 + self.Lz_coupling * Lvec[:, 2]
        energy_jf = energy_jf + self.L2_coupling * L2 + self.Lz_coupling * Lvec[:, 2]

        # Collect observables
        obs = jnp.zeros(shape=(energy.shape[0], 7), dtype=jnp.complex128)  # num walkers, num observables
        obs = obs.at[:, 0].set(energy)
        obs = obs.at[:, 1].set(energy_jf)
        obs = obs.at[:, 2].set(energy_pt)
        obs = obs.at[:, 3].set(L2)
        obs = obs.at[:, 4:7].set(Lvec)

        return obs

    def energy_compute(self, params, r, sz):
        """
        Intermediate function to return observables averaged over all walkers.
        """

        def energy_map_helper(ind, obs_stored):
            return obs_stored.at[:, ind].set(self.energy(params, r[:, ind], sz[:, ind]))

        obs_stored = jnp.zeros(shape=(r.shape[0], r.shape[1], 7), dtype=jnp.complex128)
        obs_stored = jax.lax.fori_loop(0, r.shape[1], energy_map_helper, obs_stored)
        energy = obs_stored[:, :, 0]

        obs_blk = jnp.mean(obs_stored, axis=1)
        obs_blk = pmean(obs_blk, axis_name=self.map_axis_name)

        obs_avg = jnp.mean(obs_blk, axis=0)
        obs_err = jnp.std(obs_blk, axis=0, ddof=1) / jnp.sqrt(obs_blk.shape[0])

        return (energy, obs_avg, obs_err)

    def density(self, r, sz):
        """Computes the expectation value of the single-nucleon density
        """
        if (self.remove_cm):
            rcm = jnp.mean(r, axis=1)
            r = r - rcm[:, None, :]
        r = jnp.sqrt(jnp.sum(r**2, axis=(2)))

        proton_i = (1 + sz[:, :, 1]) / 2
        neutron_i = (1 - sz[:, :, 1]) / 2

        rho_nucleon, _ = jnp.histogram(r, bins=self.nrho, range=(self.rho_min, self.rho_max), density=False)
        rho_proton, _ = jnp.histogram(r, bins=self.nrho, range=(self.rho_min, self.rho_max), density=False, weights = proton_i)
        rho_neutron, _ = jnp.histogram(r, bins=self.nrho, range=(self.rho_min, self.rho_max), density=False, weights = neutron_i)

        rho = jnp.vstack((rho_nucleon, rho_proton, rho_neutron))

        rho = rho / self.v_rho
        rho = psum(rho, axis_name=self.map_axis_name) / self.nwalk

        return rho

    def density_compute(self, r_map, sz_map):
        """Call function to compute density and compute observable values for all blocks
        """

        def density_map_helper(ind):
            return self.density(r_map[:, ind], sz_map[:, ind])

        k = jnp.arange(r_map.shape[1])
        rho = jax.lax.map(density_map_helper, k)
        rho_avg = jnp.mean(rho, axis=0)
        rho_err = jnp.std(rho, axis=0, ddof=1) / jnp.sqrt(rho.shape[0])
        rho_norm = 4 * jnp.pi * trapezoid(rho_avg * self.r_rho**2, self.r_rho)
        return rho_avg, rho_err, rho_norm

    def density_print(self, rho_avg, rho_err, rho_norm, precision=8, filename="density.dat"):

        """Prints the density on a file
        """
        data = jnp.vstack((self.r_rho, rho_avg[0, :], rho_err[0, :], rho_avg[1, :],
            rho_err[1, :], rho_avg[2, :], rho_err[2, :] )).T

        format_string = f"%.{precision}f " * 7
        np.savetxt(filename, data, fmt=format_string.strip())
        return

    def spin_isospin(self, sz):
        Sz = jnp.mean(sz[:, :, 0]) * self.npart / 2
        Tz = jnp.mean(sz[:, :, 1]) * self.npart / 2
        return Sz, Tz

    def radius(self, r, sz):
        """Computes the expectation value of the single-nucleon radius
        """
        if (self.remove_cm):
            rcm = jnp.mean(r, axis=1)
            r = r - rcm[:, None, :]
        r2 = jnp.sum(r**2, axis=2)
        proton_i = (1 + sz[:, :, 1]) / 2
        neutron_i = (1 - sz[:, :, 1]) / 2
        r2_nucleon = jnp.average(r2)
        r2_proton = jnp.average(r2, weights = proton_i)
        r2_neutron = jnp.average(r2, weights = neutron_i)
        r2 = jnp.array([r2_nucleon, r2_proton, r2_neutron])
        r2 = pmean(r2, axis_name=self.map_axis_name)
        return r2

    def radius_compute(self, r_map, sz_map):
        """Call function to compute radii and compute observable values for all blocks
        """

        def radius_map_helper(ind):
            return self.radius(r_map[:, ind], sz_map[:, ind])

        k = jnp.arange(r_map.shape[1])
        r2 = jax.lax.map(radius_map_helper, k)
        r2_avg = jnp.mean(r2, axis=0)
        r2_err = jnp.std(r2, axis=0, ddof=1) / jnp.sqrt(r2.shape[0])
        return r2_avg, r2_err

    def variance_compute(self, e_map):
        """Call function to compute variance
        """
        energy_mean = pmean(jnp.mean(e_map), axis_name=self.map_axis_name)
        energy2_mean = pmean(jnp.mean(jnp.conjugate(e_map)*e_map), axis_name=self.map_axis_name)

        variance = jnp.real(energy2_mean - jnp.abs(energy_mean)**2)
        return variance

    def angular_momentum(self, params, r, sz):
        """Computes the expectation value of the orbital angular momentum
        """
        self.dphi = 1e-4
        cos_dphi = jnp.cos(self.dphi)
        sin_dphi = jnp.sin(self.dphi)
        L2 = 0j
        Lvec = jnp.zeros(3, dtype=jnp.complex128)
        wpsi = self.wavefunction.psi( params, r, sz )
        rot_r = jnp.zeros(shape=[self.npart,3])
        if (self.remove_cm):
            rcm = jnp.mean(r, axis=0)
            r = r - rcm[None,:]
        for i in range(3):
            j = self.levi[0,i]
            k = self.levi[1,i]
            rot_r = rot_r.at[:,i].set(r[:,i])
            rot_r = rot_r.at[:,j].set(cos_dphi * r[:,j] - sin_dphi * r[:,k])
            rot_r = rot_r.at[:,k].set(cos_dphi * r[:,k] + sin_dphi * r[:,j])
            wpsi_p = self.wavefunction.psi( params, rot_r, sz )
            rot_r = rot_r.at[:,i].set(r[:,i])
            rot_r = rot_r.at[:,j].set(cos_dphi * r[:,j] + sin_dphi * r[:,k])
            rot_r = rot_r.at[:,k].set(cos_dphi * r[:,k] - sin_dphi * r[:,j])
            wpsi_m = self.wavefunction.psi( params, rot_r, sz )
            L2 = L2 - ( wpsi_p + wpsi_m - 2 * wpsi) / wpsi / self.dphi**2
            Lvec = Lvec.at[i].set( ( wpsi_p - wpsi_m ) / wpsi / 2j / self.dphi )
        return (L2, Lvec)

    def angular_momentum_vmap(self, params, r, sz):
        return vmap(self.angular_momentum, in_axes=(None, 0, 0))(params, r, sz)

    def angular_momentum_pmap(self, params, r_pmap, sz_pmap):
        return pmap(self.angular_momentum_vmap, in_axes=(None, 0, 0))(params, r_pmap, sz_pmap)

    @partial(jax.jit, static_argnums=(0,))
    def two_body_pair_singlet_triplet(self, k, params, r, sz, logpsi):
        r_ij = r[self.ip[k], :] - r[self.jp[k], :]

        if self.periodic:
            r_ij = r_ij - jnp.rint(r_ij/self.L)*self.L

        sz_ij = self.spin.sp_exch(sz, k, 0)
        logpsi_s_ij = self.wavefunction.logpsi(params, r, sz_ij)
        Ps_ij = jnp.exp(logpsi_s_ij - logpsi)

        if self.periodic:
            r_temp = r_ij + self.box_vecs
        else:
            r_temp = r_ij
        r_temp = jnp.sqrt(jnp.sum(r_temp**2, axis=-1))

        pair_dist = r_temp
        spin_weights = (2*Ps_ij-1)*jnp.ones(pair_dist.shape, dtype=jnp.complex128)

        return pair_dist, spin_weights

    @partial(jax.jit, static_argnums=(0,))
    def two_body_density_singlet_triplet(self, params, r, sz):
        """
        r is npart x ndim
        sz is npart x 2
        """
        if self.remove_cm:
            rcm = jnp.mean(r, axis=0)
            r = r - rcm

        k = jnp.arange(self.npair)
        logpsi = self.wavefunction.logpsi(params, r, sz)

        two_body_pair_map = lambda k: self.two_body_pair_singlet_triplet(k, params, r, sz, logpsi)
        # r_all, r_uu, r_dd, r_ud = jax.lax.map(vmap(two_body_pair_map), k)
        pair_dist, spin_weights = jax.lax.map(vmap(two_body_pair_map), k.reshape((-1, self.divisor)))

        pair_dist = pair_dist.reshape(-1)
        spin_weights = spin_weights.reshape(-1)

        hist_central = jnp.histogram(pair_dist, bins=self.nrho, range=(self.rho_min, self.rho_max))
        hist_spin = jnp.histogram(pair_dist, bins=self.nrho, range=(self.rho_min, self.rho_max), weights=spin_weights)

        # The division by 2 is because we count each pair only once.
        gc_r = hist_central[0]/(self.v_rho*self.npart*(self.npart/self.L**3)/2)
        gs_r = hist_spin[0]/(self.v_rho*self.npart*(self.npart/self.L**3)/2)

        g0_r = (gc_r-gs_r)/4
        g1_r = (3*gc_r+gs_r)/4

        obs = jnp.zeros(shape=[2, self.nrho], dtype=jnp.complex128)
        obs = obs.at[0, :].set(g0_r)
        obs = obs.at[1, :].set(g1_r)

        return obs

    def two_body_density_vmap_singlet_triplet(self, params, r, sz):
        return vmap(self.two_body_density_singlet_triplet, in_axes=(None, 0, 0))(params, r, sz)

    def two_body_density_pmap_singlet_triplet(self, params, r, sz):
        return pmap(self.two_body_density_vmap_singlet_triplet, in_axes=(None, 0, 0))(params, r, sz)

    @partial(jax.jit, static_argnums=(0,))
    def two_body_pair_isospin(self, k, r, sz):
        r_ij = r[self.ip[k], :] - r[self.jp[k], :]

        if self.periodic:
            r_ij = r_ij - jnp.rint(r_ij/self.L)*self.L

        proton_i = (1+sz[self.ip[k], 1])/2
        neutron_i = (1-sz[self.ip[k], 1])/2
        proton_j = (1+sz[self.jp[k], 1])/2
        neutron_j = (1-sz[self.jp[k], 1])/2

        if self.periodic:
            r_temp = r_ij + self.box_vecs
        else:
            r_temp = r_ij
        r_temp = jnp.sqrt(jnp.sum(r_temp**2, axis=-1))

        pair_dist = r_temp
        nn_proj = neutron_i*neutron_j*jnp.ones(pair_dist.shape)
        np_proj = (neutron_i*proton_j+proton_i*neutron_j)*jnp.ones(pair_dist.shape)
        pp_proj = proton_i*proton_j*jnp.ones(pair_dist.shape)

        return pair_dist, nn_proj, np_proj, pp_proj

    @partial(jax.jit, static_argnums=(0,))
    def two_body_density_isospin(self, r, sz):
        """
        r is npart x ndim
        sz is npart x 2
        """
        if self.remove_cm:
            rcm = jnp.mean(r, axis=0)
            r = r - rcm

        k = jnp.arange(self.npair)

        two_body_pair_map = lambda k: self.two_body_pair_isospin(k, r, sz)
        pair_dist, nn_proj, np_proj, pp_proj = jax.lax.map(vmap(two_body_pair_map), k.reshape((-1, self.divisor)))

        pair_dist = pair_dist.reshape(-1)
        nn_proj = nn_proj.reshape(-1)
        np_proj = np_proj.reshape(-1)
        pp_proj = pp_proj.reshape(-1)

        hist_nn = jnp.histogram(pair_dist, bins=self.nrho, range=(self.rho_min, self.rho_max), weights=nn_proj)
        hist_np = jnp.histogram(pair_dist, bins=self.nrho, range=(self.rho_min, self.rho_max), weights=np_proj)
        hist_pp = jnp.histogram(pair_dist, bins=self.nrho, range=(self.rho_min, self.rho_max), weights=pp_proj)

        gnn = hist_nn[0]/self.v_rho
        gnp = hist_np[0]/self.v_rho
        gpp = hist_pp[0]/self.v_rho

        obs = jnp.zeros(shape=[3, self.nrho], dtype=jnp.float64)
        obs = obs.at[0, :].set(gnn)
        obs = obs.at[1, :].set(gnp)
        obs = obs.at[2, :].set(gpp)

        return obs

    def two_body_density_vmap_isospin(self, r, sz):
        return vmap(self.two_body_density_isospin, in_axes=(0, 0))(r, sz)

    def two_body_density_isospin_compute(self, r, sz):

        def density_map_helper(ind):
            return self.two_body_density_vmap_isospin(r[:, ind], sz[:, ind])

        k = jnp.arange(r.shape[1])
        obs_stored = jax.lax.map(density_map_helper, k)

        obs_blk = jnp.mean(obs_stored, axis=1)
        obs_blk = pmean(obs_blk, axis_name=self.map_axis_name)

        obs_avg = jnp.mean(obs_blk, axis=0)
        obs_err = jnp.std(obs_blk, axis=0, ddof=1) / jnp.sqrt(obs_blk.shape[0])

        return (obs_avg, obs_err)
