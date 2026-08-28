import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from jax.lax import pmax, pmean, pmin

from functools import partial
import pickle
from jax.flatten_util import ravel_pytree

from WavefunctionUtil import WavefunctionUtil
from Backflow import Backflow
from Asymm import Asymm


class Wavefunction(object):
    def __init__(self, config, mesh, parity=None):
        self.util = WavefunctionUtil(config)

        if parity is None:
            parity = config.parity

        asymm_name = config.asymm_type
        backflow_name = config.backflow_type

        self.npart = config.npart
        self.conf = config.conf
        self.parity = parity
        self.mix = config.mix

        self.R0 = 1.2 * jnp.cbrt(self.npart) / jnp.sqrt( 3 )
        self.remove_cm = config.remove_cm
        self.periodic = config.periodic
        self.L = config.L

        self.backflow = Backflow(backflow_name, self.util, config)
        bf_out_size = (self.backflow.single_size, self.backflow.paired_size)
        self.asymm = Asymm(asymm_name, bf_out_size, self.util, config)

        self.vmap_psi = jax.jit(self.vmap_psi)

        self.map_axis_name = mesh.axis_names[0]
        Pspec = P(self.map_axis_name)
        Pnone = P()

        self.shard_map_logpsi = shard_map(self.vmap_logpsi, mesh=mesh,
                                          in_specs=(Pnone, Pspec, Pspec),
                                          out_specs=(Pspec),
                                          check_rep=False)

        self.shard_map_logpsi = jax.jit(self.shard_map_logpsi)

    def build(self):
        bf_params = self.backflow.build()
        asymm_params = self.asymm.build()

        net_params = [bf_params, asymm_params]

        # cast to double
        net_params = jax.tree_util.tree_map(self.update_cast, net_params)
        flat_net_params = self.flatten_params(net_params)
        num_flat_params = flat_net_params.shape[0]

        with open('full.model', 'wb') as file:
            pickle.dump(net_params, file)

        return net_params, num_flat_params

    @partial(jit, static_argnums=(0,))
    def logphasepsi(self, params, r, sz):
        """
        r is npart x ndim
        sz is npart x 2
        """
        # Separate parameters
        bf_params, asymm_params = params

        if self.remove_cm:
            r = self.util.remove_cm_util(r)

        if not self.periodic:
            r = r/self.R0
        else:
            r = r - jnp.rint(r/self.L)*self.L

        # Modify hidden coordinate input
        x, x_pair = self.backflow.run(bf_params, r, sz)

        # Pass through Antisymmetric layer
        phase, log = self.asymm.run(asymm_params, x, x_pair)

        # Nuclei confinement
        if not self.periodic:
            log = log - self.conf * jnp.sum(r**2)

        phase = jnp.reshape(phase, ())
        log = jnp.reshape(log, ())

        return phase, log

    @partial(jit, static_argnums=(0,))
    def logpsi(self, params, r, sz):
        sz_pm = jnp.stack((sz, sz))
        r_pm = jnp.stack((r, -r))
        sign, log = vmap(self.logphasepsi, in_axes=(None, 0, 0))(params, r_pm, sz_pm)
        log = log + 1j * sign
        p_sign = jnp.asarray([1, self.parity])
        log = jax.scipy.special.logsumexp(a=log, b=p_sign, return_sign=False)
        return log

    @partial(jit, static_argnums=(0,))
    def psi(self, params, r, sz):
        log = self.logpsi(params, r, sz)
        return jnp.exp(log)

    def vmap_psi(self, params, r_batched, sz_batched):
        return vmap(self.psi, in_axes=(None, 0, 0))(params, r_batched, sz_batched)

    @partial(jit, static_argnums=(0,))
    def logpsi_stats(self, params, r_batched, sz_batched):
        logpsi = self.vmap_logpsi(params, r_batched, sz_batched)
        max_logpsi = pmax(jnp.max(jnp.real(logpsi)), axis_name=self.map_axis_name)
        avg_logpsi = pmean(jnp.mean(jnp.real(logpsi)), axis_name=self.map_axis_name)
        min_logpsi = pmin(jnp.min(jnp.real(logpsi)), axis_name=self.map_axis_name)
        return logpsi, max_logpsi, avg_logpsi, min_logpsi

    @partial(jit, static_argnums=(0,))
    def vmap_logpsi(self, params, r_batched, sz_batched):
        return vmap(self.logpsi, in_axes=(None, 0, 0))(params, r_batched, sz_batched)

    @partial(jit, static_argnums=(0,))
    def dlogpsi(self, params, r, sz):
        dlogpsi_dx = jax.grad(self.logpsi, argnums=1)(params, r, sz)
        return dlogpsi_dx

    @partial(jit, static_argnums=(0,))
    def vmap_dlogpsi(self, params, r_batched, sz_batched):
        return vmap(self.dlogpsi, in_axes=(None, 0, 0))(params, r_batched, sz_batched)

    @partial(jit, static_argnums=(0,))
    def vmap_psi_sum(self, params, r_batched, sz_batched):
        return jnp.sum(self.vmap_psi(params, r_batched, sz_batched))

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

    @partial(jit, static_argnums=(0,))
    def update_subtract(self, params, dparams):
        return params - dparams

    @partial(jit, static_argnums=(0,))
    def update_mix(self, params, dparams):
        return self.mix * params + (1 - self.mix) * dparams

    @partial(jit, static_argnums=(0,))
    def update_cast(self, params):
        return params.astype(jnp.float64)

    @partial(jit, static_argnums=(0,))
    def update_zero(self, params):
        return 0 * params
