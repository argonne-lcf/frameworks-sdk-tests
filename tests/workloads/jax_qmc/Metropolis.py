import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from jax.experimental.multihost_utils import host_local_array_to_global_array, process_allgather
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.lax import fori_loop, pmean
from functools import partial


class Metropolis(object):
    """Metropolis Sampler in N dimension

    Sample from N-D coordinates, using the M(RT)^2 algorithm.
    It starts from a Gaussian distribution,  then propagates
    for nvoid steps nobs times, and stores the walk

    """
    def __init__(self, n_devices, wavefunction, config, logger, mesh):
        self.recover_r = config.rec_r
        self.recover_k = config.rec_k
        self.n_devices = n_devices
        self.nvoid = config.nvoid
        self.neq = config.neq
        self.nav = config.nav
        self.nac = config.nac
        self.nsteps = (self.neq + self.nav + self.nac) * self.nvoid
        self.nwalk = config.nwalk
        self.npart = config.npart
        self.nprot = config.nprot
        self.nup = config.nup
        self.ndim = config.ndim
        self.conserved_sz = config.conserved_sz
        self.wavefunction = wavefunction

        self.R0 = 1.2 * jnp.cbrt(self.npart) / jnp.sqrt(3)
        self.remove_cm = config.remove_cm
        self.periodic = config.periodic
        self.L = config.L

        self.npair = int(self.npart * (self.npart - 1) / 2)
        self.ip = jnp.empty(self.npair, dtype=int)
        self.jp = jnp.empty(self.npair, dtype=int)
        k = 0
        for i in range(self.npart-1):
            for j in range(i+1, self.npart):
                self.ip = self.ip.at[k].set(i)
                self.jp = self.jp.at[k].set(j)
                k += 1

        self.key = random.PRNGKey(config.seed_walk)
        mass = (config.mass_p+config.mass_n)/2
        self.sigma = jnp.sqrt(config.hbar**2*config.dt/mass)

        print('self.sigma=', self.sigma)

        self.logger = logger
        self.walk = jax.jit(self.walk)
        self.vmap_walk = jax.jit(self.vmap_walk)

        # prep shardmap
        self.map_axis_name = mesh.axis_names[0]
        self.mesh = mesh
        Pspec = P(self.map_axis_name)
        Pnone = P()

        self.shard_map_walk = shard_map(self.vmap_walk, mesh=self.mesh,
                                        in_specs=(Pnone, Pspec, Pspec, Pspec),
                                        out_specs=(Pspec, Pspec, Pnone, Pspec))
        self.shard_map_walk = jax.jit(self.shard_map_walk)

    # Spin or isospin exchange operation
    @partial(jit, static_argnums=(0,))
    def sp_exch(self, sz_o, k, ist):
        sz_n = sz_o.at[self.ip[k], ist].set(sz_o[self.jp[k], ist])
        sz_n = sz_n.at[self.jp[k], ist].set(sz_o[self.ip[k], ist])
        return sz_n

    # Spin propagation by flipping the spin and the isospin of pairs ks and kt
    @partial(jit, static_argnums=(0,))
    def sp_prop_old(self, sz_o, ks, kt):
        sz_n = self.sp_exch(sz_o, ks, 0)
        sz_n = self.sp_exch(sz_n, kt, 1)
        return sz_n

    @partial(jit, static_argnums=(0,))
    def sp_prop_permute(self, sz_o, k, key):
        i_flip = jax.random.randint(key, (), -1, 2)

        # Perform the exchange depending on the value of i_flip:
        # - Only spin if i_flip == -1
        # - Spin and isospin if i_flip == 0
        # - Only isospin if i_flip == 1

        # Exchange only spin (i_rand == -1)
        sz_spin_only = self.sp_exch(sz_o, k, 0)

        # Exchange both spin and isospin (i_rand == 0)
        sz_both = self.sp_exch(sz_spin_only, k, 1)

        # Exchange only isospin (i_rand == 1)
        sz_isospin_only = self.sp_exch(sz_o, k, 1)

        # Select the appropriate result based on i_rand
        sz_n = jax.lax.select(i_flip == -1, sz_spin_only, jax.lax.select(i_flip == 0, sz_both, sz_isospin_only))

        return sz_n

    # Spin propagation by flipping the spin and the isospin of pairs ks and kt
    @partial(jit, static_argnums=(0,))
    def sp_prop_flip(self, sz_o, ks, kt, key):
        sz_n = self.sp_flip(sz_o, ks, 0, key)
        sz_n = self.sp_exch(sz_n, kt, 1)
        return sz_n

    # Spin or isospin exchange operation
    @partial(jit, static_argnums=(0,))
    def sp_flip(self, sz_o, k, ist, key):
        sz_flipped = 2 * jax.random.randint(key, shape=[2], minval=0, maxval=2) - 1
        sz_n = sz_o.at[self.ip[k], ist].set(sz_flipped[0])
        sz_n = sz_n.at[self.jp[k], ist].set(sz_flipped[1])
        return sz_n

    # Note: sp_permute and sp_permute_fast are not used currently but may be
    # soon. They ensure the proposed spin update is always different from the
    # original.
    @partial(jit, static_argnums=(0,))
    def sp_permute(self, sz_o, key):
        # Permute isospins and spins of a random set such that it will be
        # different from the original.
        subset_size = 2

        # Select subset for which swaps can change results
        def cond_fun1(loop_carry):
            key, subset = loop_carry
            test = sz_o[subset, :]
            identical_spin = jnp.all(test[:, 0] == test[0, 0])
            identical_isospin = jnp.all(test[:, 1] == test[0, 1])
            return jnp.logical_and(identical_spin, identical_isospin)

        def body_fun1(loop_carry):
            key, subset = loop_carry
            key, key_input = jax.random.split(key)
            subset = random.choice(key_input, jnp.arange(self.npart), shape=(subset_size,), replace=False)
            return key, subset

        key, subset = jax.lax.while_loop(cond_fun1, body_fun1, (key, jnp.zeros(shape=subset_size, dtype=jnp.int64)))

        def cond_fun2(loop_carry):
            key, sz_n = loop_carry
            return jnp.array_equal(sz_o, sz_n)

        def body_fun2(loop_carry):
            key, sz_n = loop_carry
            key, key_input = jax.random.split(key)
            sz_n = sz_n.at[subset, :].set(random.permutation(key_input, sz_o[subset, :], 0, independent=True))
            return key, sz_n

        key, sz_n = jax.lax.while_loop(cond_fun2, body_fun2, (key, sz_o))

        return key, sz_n

    @partial(jit, static_argnums=(0,))
    def sp_permute_fast(self, sz_o, key):
        # Permute isospins and spins of a random pair such that it will be
        # different from the original.

        # Select subset for which swaps can change results
        def cond_fun1(loop_carry):
            key, k = loop_carry
            identical_spin = sz_o[self.ip[k], 0] == sz_o[self.jp[k], 0]
            identical_isospin = sz_o[self.ip[k], 1] == sz_o[self.jp[k], 1]
            return jnp.reshape(jnp.logical_and(identical_spin, identical_isospin), ())

        def body_fun1(loop_carry):
            key, k = loop_carry
            key, key_input = jax.random.split(key)
            k = jax.random.randint(key_input, shape=(1,), minval=0, maxval=self.npair)
            return key, k

        key, key_input = jax.random.split(key)
        k = jax.random.randint(key_input, shape=(1,), minval=0, maxval=self.npair)
        key, k = jax.lax.while_loop(cond_fun1, body_fun1, (key, k))

        key, key_input = random.split(key)
        i_rand = jax.random.randint(key_input, (), -1, 2)
        # Exchange only spin (i_rand == -1)
        sz_spin_only = self.sp_exch(sz_o, k, 0)
        # Exchange both spin and isospin (i_rand == 0)
        sz_both = self.sp_exch(sz_spin_only, k, 1)
        # Exchange only isospin (i_rand == 1)
        sz_isospin_only = self.sp_exch(sz_o, k, 1)
        # Select the appropriate result based on i_rand
        sz_rand = jax.lax.select(i_rand == -1, sz_spin_only, jax.lax.select(i_rand == 0, sz_both, sz_isospin_only))

        # We removed the possibility of both spin and isospin being identical
        # with the previous while loop. Now if ether is identical we can return
        # both swapped to get the only new spin state for this pair.
        # Otherwise we need to randomly determine which to swap.
        identical_spin = sz_o[self.ip[k], 0] == sz_o[self.jp[k], 0]
        identical_isospin = sz_o[self.ip[k], 1] == sz_o[self.jp[k], 1]
        selector = jnp.reshape(jnp.logical_or(identical_spin, identical_isospin), ())
        sz_n = jax.lax.select(selector, sz_both, sz_rand)

        return key, sz_n

    # Initialize the spin walk making sure the wave function is different from zero
    @partial(jit, static_argnums=(0, 2))
    def sp_init(self, key, wavefunction, params, r):
        def cond_fun(loop_carry):
            key, sz_o = loop_carry
            wphi_o = wavefunction.vmap_psi(params, r, sz_o)
            return jnp.min(jnp.abs(wphi_o)) == 0

        def body_fun(loop_carry):
            key, sz_o = loop_carry
            key, key_input = jax.random.split(key)
            sz_p = random.permutation(key_input, sz_o, 1, independent=True)
            wphi_p = wavefunction.vmap_psi(params, r, sz_p)
            accept = (wphi_p != 0)
            sz_o = jnp.where(accept.reshape([self.nwalk, 1, 1]), sz_p, sz_o)
            return key, sz_o

        sz_o = -1 * jnp.ones(shape=(self.nwalk, self.npart, 2))
        sz_o = sz_o.at[:, 0:self.nup, 0].set(1)
        sz_o = sz_o.at[:, 0:self.nprot, 1].set(1)
        key, key_input = random.split(key)
        sz_o = random.permutation(key_input, sz_o, 1, independent=True)
        key, sz_o = jax.lax.while_loop(cond_fun, body_fun, (key, sz_o))
        return sz_o

    def recover_samples(self, key_input, r_s, sz_s, logpsi_s, logpsi_n):
        logpsi_s_block = process_allgather(logpsi_s, tiled=True)
        logpsi_n_block = process_allgather(logpsi_n, tiled=True)
        r_o_block = process_allgather(r_s, tiled=True)
        sz_o_block = process_allgather(sz_s, tiled=True)

        weights = jnp.abs(jnp.exp(2 * (logpsi_n_block - logpsi_s_block)))
        weights = weights / jnp.sum(weights)

        indices_block = jax.random.choice(key_input, logpsi_s_block.shape[0], shape=(self.nwalk,), replace=False, p=weights)

        r_o = r_o_block[indices_block, :, :].reshape((jax.process_count(), self.nwalk//jax.process_count(), self.npart, self.ndim))
        r_o = host_local_array_to_global_array(r_o[jax.process_index()], self.mesh, P(self.map_axis_name))
        sz_o = sz_o_block[indices_block, :, :].reshape((jax.process_count(), self.nwalk//jax.process_count(), self.npart, 2))
        sz_o = host_local_array_to_global_array(sz_o[jax.process_index()], self.mesh, P(self.map_axis_name))

        return r_o, sz_o

    def initialize(self, it, params):
        """
        Creates initial random distributions of nwalk 'walkers' which are the
        particle coordinates and spins for npart particles. Along with this a
        separate key is created for each walker for generating random numbers.

        r_o: array with shape (nwalk, npart, ndim)
        sz_o: array with shape (nwalk, npart, 2)
        key_o: array with shape (nwalk,)
        """
        if (self.recover_k and it > 0):
            key = self.key_s
        else:
            key = self.key
        if (self.recover_r and it > 0):
            key, key_input = jax.random.split(key)
            r_s = self.r_s
            sz_s = self.sz_s
            logpsi_s = self.logpsi_s
            logpsi_n = self.wavefunction.shard_map_logpsi(params, r_s, sz_s)
            r_o, sz_o = self.recover_samples(key_input, r_s, sz_s, logpsi_s, logpsi_n)
        else:
            key, key_input = jax.random.split(key)
            if not self.periodic:
                r_o = self.R0 * jax.random.normal(key_input, shape=[self.nwalk, self.npart, self.ndim])
            else:
                r_o = self.L * jax.random.uniform(key_input, shape=[self.nwalk, self.npart, self.ndim]) - self.L/2
                r_o = r_o - jnp.rint(r_o/self.L)*self.L

            if (self.remove_cm):
                rcm = jnp.mean(r_o, axis=1)
                r_o = r_o - rcm[:, None, :]

            #r_o = r_o.reshape((jax.process_count(), self.nwalk//jax.process_count(), self.npart, self.ndim))
            #r_o = host_local_array_to_global_array(r_o[jax.process_index()], self.mesh, P(self.map_axis_name))

            key, key_input = jax.random.split(key)
            sz_o = self.sp_init(key_input, self.wavefunction, params, r_o)

        key, key_o = jax.random.split(key)
        key_o = jax.random.split(key_o, self.nwalk)
        self.key_s = key

        key_o = key_o.reshape((jax.process_count(), self.nwalk//jax.process_count(), -1))
        key_o = host_local_array_to_global_array(key_o[jax.process_index()], self.mesh, P(self.map_axis_name))

        return key_o, r_o, sz_o

# Performs the Metropolis walk using the fori loop constructions
    def walk(self, params, r_o, sz_o, key_o):

        # Single step fori_loop construct
        def step(i, loop_carry_i):
            r_o, sz_o, key_o, logpsi_o, acc_s, r_s, sz_s, logpsi_s = loop_carry_i

            # Move coordinates with drift and reweight
            key_o, key_input = random.split(key_o)
            r_gauss = self.sigma * jax.random.normal(key_input, shape=[self.npart, self.ndim])
            r_n = r_o + r_gauss
            rcm = jnp.mean(r_n, axis=0)
            r_n = r_n - rcm[None, :]
#            r_n = 40. * jnp.tanh(r_n / 40.)
            r_n = jnp.clip(r_n, -15, 15)

            logpsi_n = self.wavefunction.logpsi(params, r_n, sz_o)
            prob = jnp.abs(jnp.exp(2 * (logpsi_n - logpsi_o)))
            key_o, key_input = random.split(key_o)
            unif = jax.random.uniform(key_input)
            accept = jnp.greater_equal(prob, unif)
#            accept = jnp.greater_equal(prob, 0)
            r_o = jnp.where(accept.reshape([1,1]), r_n, r_o)
            logpsi_o = jnp.where(accept, logpsi_n, logpsi_o)
            acc_s = acc_s.at[i // self.nvoid,0].set(accept.astype('float64'))

            # Swap spin, isospin, or both of pair ks
            key_o, key_input = random.split(key_o)
            ks = jax.random.randint(key_input, shape=[1], minval=0, maxval=self.npair)
            key_o, key_input = random.split(key_o)
            if (self.conserved_sz):
                sz_n = self.sp_prop_permute(sz_o, ks, key_input)
            else:
                kt = jax.random.randint(key_input, shape=[1], minval=0, maxval=self.npair)
                key_o, key_input = random.split(key_o)
                sz_n = self.sp_prop_flip(sz_o, ks, kt, key_input)
            logpsi_n = self.wavefunction.logpsi(params, r_o, sz_n)
            prob = jnp.abs(jnp.exp(2 * (logpsi_n - logpsi_o)))
            key_o, key_input = random.split(key_o)
            unif = jax.random.uniform(key_input)
            accept = jnp.greater_equal(prob, unif)
            sz_o = jnp.where(accept.reshape([1, 1]), sz_n, sz_o)
            logpsi_o = jnp.where(accept, logpsi_n, logpsi_o)
            acc_s = acc_s.at[i // self.nvoid, 1].set(accept.astype('float64'))

            r_s = r_s.at[i // self.nvoid, :, :].set(r_o)
            sz_s = sz_s.at[i // self.nvoid, :, :].set(sz_o)
            logpsi_s = logpsi_s.at[i // self.nvoid].set(logpsi_o)

            return r_o, sz_o, key_o, logpsi_o, acc_s, r_s, sz_s, logpsi_s

        if (self.remove_cm):
            rcm = jnp.mean(r_o, axis=0)
            r_o = r_o - rcm[None, :]
#            r_o = 40. * jnp.tanh(r_o / 40.)
            r_o = jnp.clip(r_o, -15, 15)


        acc_s = jnp.zeros(shape=[self.neq + self.nav + self.nac, 2])
        r_s = jnp.zeros(shape=[self.neq + self.nav + self.nac, self.npart, self.ndim])
        sz_s = jnp.zeros(shape=[self.neq + self.nav + self.nac, self.npart, 2])
        logpsi_s = jnp.zeros(shape=[self.neq + self.nav + self.nac], dtype=jnp.complex128)

        logpsi_o = self.wavefunction.logpsi(params, r_o, sz_o)
        r_o, sz_o, key_o, logpsi_o, acc_s, r_s, sz_s, logpsi_s = fori_loop(0, self.nsteps, step, (r_o, sz_o, key_o, logpsi_o, acc_s, r_s, sz_s, logpsi_s) )

        acc_s = acc_s[self.neq:, :]
        r_s = r_s[self.neq:, :, :]
        sz_s = sz_s[self.neq:, :, :]
        logpsi_s = logpsi_s[self.neq:]

        if self.periodic:
            r_s = r_s - jnp.rint(r_s/self.L)*self.L

        return r_s, sz_s, acc_s, logpsi_s

    def vmap_walk(self, params, r_o, sz_o, key_o):
        r_s, sz_s, acc_s, logpsi_s = vmap(self.walk, in_axes=(None, 0, 0, 0))(params, r_o, sz_o, key_o)
        acc_s = jnp.mean(acc_s, axis=(0, 1))
        acc_s = pmean(acc_s, axis_name=self.map_axis_name)
        return r_s, sz_s, acc_s, logpsi_s

    def shmap_walk(self, params, r_o, sz_o, key_o):
        """
        Preform a Metropolis walk for each walker, retaining nav samples from
        each walk. Subsequently we have nwalk*nav samples, however the walkers
        have been split over the number of devices we are using. So we do not
        have access to the entire array without going back into a sharded
        function. The return values have sizes
        r_map: (nwalk, nav, npart, ndim)
        sz_map: (nwalk, nav, npart, 2)
        acc is an array of two values (2,)
        """
        r_map, sz_map, acc, logpsi_map = self.shard_map_walk(params, r_o, sz_o, key_o)

#        self.r_s = r_map[:, self.nav + self.nac - 1, :, :]
#        self.sz_s = sz_map[:, self.nav + self.nac - 1, :, :]
#        self.logpsi_s = logpsi_map[:, self.nav + self.nac - 1]

        self.r_s = jnp.reshape(r_map, (r_map.shape[0] * r_map.shape[1],) + r_map.shape[2:])
        self.sz_s = jnp.reshape(sz_map, (sz_map.shape[0] * sz_map.shape[1],) + sz_map.shape[2:])
        self.logpsi_s = jnp.reshape(logpsi_map, (logpsi_map.shape[0] * logpsi_map.shape[1],))

        acc_r = acc[0]
        acc_sz = acc[1]
        return r_map, sz_map, acc_r, acc_sz
