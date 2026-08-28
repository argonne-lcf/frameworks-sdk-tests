import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial


class Backflow(object):
    def __init__(self, name, util, config):
        self.util = util

        self.periodic = config.periodic
        self.L = config.L

        self.npart = config.npart
        self.ndim = config.ndim
        self.ndense = config.ndense
        if name == 'MPNN':
            self.dense_depth = config.mpnn_dense_depth
            self.gn_out_size = config.mpnn_output_size
            self.gn_layers = config.mpnn_layers

            if not self.periodic:
                self.nx_i = self.ndim + 2
                self.nx_ij = self.ndim + 5
            else:
                self.nx_i = 2
                self.nx_ij = 2*self.ndim + 5
            self.nh_i = self.nx_i + self.gn_out_size
            self.nh_ij = self.nx_ij + self.gn_out_size
            self.nm_ij = 2*self.nh_i + self.nh_ij

            # Variables and functions called outside this class
            self.build = self.build_MPNN
            self.run = self.MPNN
            self.single_size = self.nm_ij
            self.paired_size = self.nm_ij
        elif name == 'minimal_MPNN':
            self.dense_depth = config.minimal_mpnn_dense_depth
            self.nlat = config.nlat

            if not self.periodic:
                self.net_out_size = self.nlat - (self.ndim+2)
                self.nm_ij = 2*(self.ndim+2)
            else:
                self.net_out_size = self.nlat - (2*self.ndim+2)
                self.nm_ij = 2*(2*self.ndim+2)

            self.build = self.build_minimal_MPNN
            self.run = self.minimal_MPNN
            self.single_size = self.nlat
            self.paired_size = 2*(self.single_size)
        elif name in ['one_body_transformer', 'two_body_transformer']:
            self.num_heads = config.transformer_num_heads
            self.hidden_size = config.transformer_hidden_size

            self.qk_size = self.hidden_size//self.num_heads

            self.tr_layers = config.transformer_layers
            self.out_size = config.transformer_output_size

            if name == 'one_body_transformer':
                if not self.periodic:
                    self.nx_i = self.ndim + 2
                else:
                    self.nx_i = 2*self.ndim + 2

                self.build = self.build_one_body_transformer
                self.run = self.one_body_transformer
                self.single_size = self.nx_i+self.out_size
                self.paired_size = 2*(self.single_size)
            elif name == 'two_body_transformer':
                if not self.periodic:
                    self.nx_ij = self.ndim + 5
                else:
                    self.nx_ij = 2*self.ndim + 5

                self.build = self.build_two_body_transformer
                self.run = self.two_body_transformer
                self.single_size = self.nx_ij+self.out_size
                self.paired_size = self.nx_ij+self.out_size
        elif name is None or name.lower() in ['', 'none', 'no', 'false']:
            # No backflow
            self.build = lambda: []
            self.run = self.no_backflow

            if not self.periodic:
                self.single_size = self.ndim+2
            else:
                self.single_size = 2*self.ndim+2
            self.paired_size = 2*(self.single_size)
        else:
            raise NameError(f'Undefined backflow type: {name}')

        self.npair_inej = self.npart * (self.npart - 1)
        self.k_inej = jnp.arange(self.npair_inej)
        self.ip_inej = jnp.empty(self.npair_inej, dtype=int)
        self.jp_inej = jnp.empty(self.npair_inej, dtype=int)
        k = 0
        for i in range(self.npart):
            for j in range(self.npart):
                if (i != j):
                    self.ip_inej = self.ip_inej.at[k].set(i)
                    self.jp_inej = self.jp_inej.at[k].set(j)
                    k += 1

    @partial(jit, static_argnums=(0,))
    def x_pair(self, k, x):
        x_ij = jnp.concatenate((x[self.ip_inej[k], :], x[self.jp_inej[k], :]))
        return x_ij

    @partial(jit, static_argnums=(0,))
    def no_backflow(self, empty_params, r, sz):
        if not self.periodic:
            x_i = jnp.concatenate((r, sz), axis=1)
        else:
            C = jnp.cos((2*jnp.pi/self.L)*r)
            S = jnp.sin((2*jnp.pi/self.L)*r)
            x_i = jnp.concatenate((C, S, sz), axis=1)

        x_ij = vmap(self.x_pair, in_axes=(0, None))(self.k_inej, x_i)
        return x_i, x_ij

    def build_MPNN(self):
        gn_init, self.gn_apply = self.util.dense_net(
            self.ndense, self.dense_depth, self.gn_out_size
            )

        A_params = self.util.init_params(gn_init, self.nx_i)
        B_params = self.util.init_params(gn_init, self.nx_ij)
        M_params = self.util.init_params(gn_init, self.nm_ij, count=self.gn_layers)
        F_params = self.util.init_params(gn_init, self.nh_i+self.gn_out_size, count=self.gn_layers)
        G_params = self.util.init_params(gn_init, self.nh_ij+self.gn_out_size, count=self.gn_layers)

        return [A_params, B_params, M_params, F_params, G_params]

    @partial(jit, static_argnums=(0,))
    def MPNN(self, gn_params, r, sz):
        A_params, B_params, M_params, F_params, G_params = gn_params

        # initial particle features
        if not self.periodic:
            x_i = jnp.concatenate((r, sz), axis=1)

            # initial pair features
            def init_pairs(k):
                r_ij = r[self.ip_inej[k]]-r[self.jp_inej[k]]
                d_ij = jnp.sqrt(jnp.sum(r_ij**2))
                x_ij = jnp.array([d_ij, sz[self.ip_inej[k], 0], sz[self.ip_inej[k], 1], sz[self.jp_inej[k], 0], sz[self.jp_inej[k], 1]])
                x_ij = jnp.concatenate((r_ij, x_ij))
                return x_ij
        else:
            x_i = sz

            # initial pair features
            def init_pairs(k):
                r_ij = r[self.ip_inej[k]]-r[self.jp_inej[k]]
                r_ij = r_ij - jnp.rint(r_ij/self.L)*self.L
                S_ij = jnp.sin(2*jnp.pi*r_ij/self.L)
                C_ij = jnp.cos(2*jnp.pi*r_ij/self.L)
                d_ij = jnp.sin(jnp.pi*r_ij/self.L)
                d_ij = jnp.sqrt(jnp.sum(d_ij**2))
                x_ij = jnp.array([d_ij, sz[self.ip_inej[k], 0], sz[self.ip_inej[k], 1], sz[self.jp_inej[k], 0], sz[self.jp_inej[k], 1]])
                x_ij = jnp.concatenate((S_ij, C_ij, x_ij))
                return x_ij

        x_ij = vmap(init_pairs, in_axes=(0))(self.k_inej)

        # convenient function that combines one- and two-body info
        def combine_streams(k, h_i, h_ij):
            return jnp.concatenate((h_i[self.ip_inej[k]], h_i[self.jp_inej[k]], h_ij[k]))

        # initial hidden features
        h_i = self.gn_apply(A_params, x_i)
        h_i = jnp.concatenate((x_i, h_i), axis=1)
        h_ij = self.gn_apply(B_params, x_ij)
        h_ij = jnp.concatenate((x_ij, h_ij), axis=1)

        for t in range(self.gn_layers):
            m_ij = vmap(combine_streams, in_axes=(0, None, None))(self.k_inej, h_i, h_ij)
            m_ij = self.gn_apply(M_params[t], m_ij)
            h_ij = jnp.concatenate((h_ij, m_ij), axis=1)
            h_ij = self.gn_apply(G_params[t], h_ij)
            h_ij = jnp.concatenate((x_ij, h_ij), axis=1)
            m_ij = jnp.reshape(m_ij, (self.npart, self.npart-1, -1))

            m_i = jax.scipy.special.logsumexp(m_ij, axis=1)
            h_i = jnp.concatenate((h_i, m_i), axis=1)
            h_i = self.gn_apply(F_params[t], h_i)
            h_i = jnp.concatenate((x_i, h_i), axis=1)

        gn_ij = vmap(combine_streams, in_axes=(0, None, None))(self.k_inej, h_i, h_ij)

        gn_ij_reshape = jnp.reshape(gn_ij, shape=(self.npart, self.npart-1, -1))
        # gn_i = jnp.concatenate((x_i, jax.scipy.special.logsumexp(gn_ij, axis=1)), axis=1)
        gn_i = jax.scipy.special.logsumexp(gn_ij_reshape, axis=1)

        return gn_i, gn_ij

    def build_minimal_MPNN(self):
        m_init, self.m_apply = self.util.dense_net(
            self.ndense, self.dense_depth, self.net_out_size
            )

        m_params = self.util.init_params(m_init, self.nm_ij)
        return m_params

    @partial(jit, static_argnums=(0,))
    def minimal_MPNN(self, m_params, r, sz):
        if not self.periodic:
            x = jnp.concatenate((r, sz), axis=1)
        else:
            C = jnp.cos((2*jnp.pi/self.L)*r)
            S = jnp.sin((2*jnp.pi/self.L)*r)
            x = jnp.concatenate((C, S, sz), axis=1)

        x_ij = vmap(self.x_pair, in_axes=(0, None))(self.k_inej, x)
        m_ij = jnp.reshape(self.m_apply(m_params, x_ij), shape=(self.npart, self.npart-1, -1))
        new_x_i = jnp.concatenate((x, jnp.mean(m_ij, axis=1)), axis=1)
        new_x_ij = vmap(self.x_pair, in_axes=(0, None))(self.k_inej, new_x_i)

        return new_x_i, new_x_ij

    def build_one_body_transformer(self):
        tr_init, self.tr_apply = self.util.full_transformer(
            self.num_heads, self.hidden_size, self.qk_size,
            self.tr_layers, self.out_size
            )

        tr_params = self.util.init_params(tr_init, (self.npart, self.nx_i))
        return tr_params

    @partial(jit, static_argnums=(0,))
    def one_body_transformer(self, tr_params, r, sz):
        if not self.periodic:
            x = jnp.concatenate((r, sz), axis=1)
        else:
            C = jnp.cos((2*jnp.pi/self.L)*r)
            S = jnp.sin((2*jnp.pi/self.L)*r)
            x = jnp.concatenate((C, S, sz), axis=1)

        tr = self.tr_apply(tr_params, x)
        new_x_i = jnp.concatenate((x, tr), axis=1)
        new_x_ij = vmap(self.x_pair, in_axes=(0, None))(self.k_inej, new_x_i)

        return new_x_i, new_x_ij

    def build_two_body_transformer(self):
        tr_init, self.tr_apply = self.util.full_transformer(
            self.num_heads, self.hidden_size, self.qk_size,
            self.tr_layers, self.out_size
            )

        tr_params = self.util.init_params(tr_init, (self.npair_inej, self.nx_ij))
        return tr_params

    @partial(jit, static_argnums=(0,))
    def two_body_transformer(self, tr_params, r, sz):
        # initial particle features
        if not self.periodic:
            def init_pairs(k):
                r_ij = r[self.ip_inej[k]]-r[self.jp_inej[k]]
                d_ij = jnp.sqrt(jnp.sum(r_ij**2))
                x_ij = jnp.array([d_ij, sz[self.ip_inej[k], 0], sz[self.ip_inej[k], 1], sz[self.jp_inej[k], 0], sz[self.jp_inej[k], 1]])
                x_ij = jnp.concatenate((r_ij, x_ij))
                return x_ij
        else:
            def init_pairs(k):
                r_ij = r[self.ip_inej[k]]-r[self.jp_inej[k]]
                r_ij = r_ij - jnp.rint(r_ij/self.L)*self.L
                S_ij = jnp.sin(2*jnp.pi*r_ij/self.L)
                C_ij = jnp.cos(2*jnp.pi*r_ij/self.L)
                d_ij = jnp.sin(jnp.pi*r_ij/self.L)
                d_ij = jnp.sqrt(jnp.sum(d_ij**2))
                x_ij = jnp.array([d_ij, sz[self.ip_inej[k], 0], sz[self.ip_inej[k], 1], sz[self.jp_inej[k], 0], sz[self.jp_inej[k], 1]])
                x_ij = jnp.concatenate((S_ij, C_ij, x_ij))
                return x_ij

        x_ij = vmap(init_pairs, in_axes=(0))(self.k_inej)

        tr = self.tr_apply(tr_params, x_ij)
        new_x_ij = jnp.concatenate((x_ij, tr), axis=1)

        new_x_ij_reshape = jnp.reshape(new_x_ij, shape=(self.npart, self.npart-1, -1))
        # gn_i = jnp.concatenate((x_i, jax.scipy.special.logsumexp(gn_ij, axis=1)), axis=1)
        new_x_i = jax.scipy.special.logsumexp(new_x_ij_reshape, axis=1)

        return new_x_i, new_x_ij
