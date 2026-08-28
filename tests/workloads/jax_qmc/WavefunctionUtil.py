import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.nn.initializers import glorot_normal
from jax.example_libraries import stax
from jax.example_libraries.stax import (Dense, FanOut, FanInConcat, FanInSum)
from jax.example_libraries.stax import (Tanh, elementwise)
from functools import partial


def Linear_map(out_dim, W_init=glorot_normal()):
    """
    Layer constructor function for a Linear layer.
    Fully connected layer with no bias.
    """
    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        W = W_init(rng, (input_shape[-1], out_dim))
        return output_shape, W
    def apply_fun(params, inputs, **kwargs):
        W = params
        return jnp.dot(inputs, W)
    return init_fun, apply_fun


def LayerNorm(eps: float = 1e-5):
    """LayerNorm over the last axis with learnable scale (γ) and shift (β)."""
    def init_fun(rng, input_shape):
        features = input_shape[-1]
        gamma = jnp.ones(features)
        beta = jnp.zeros(features)
        return input_shape, (gamma, beta)

    def apply_fun(params, x, **kwargs):
        gamma, beta = params
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / jnp.sqrt(var + eps) + beta
    return init_fun, apply_fun


def residual_block(layer):
    """Wraps given layer in residual"""
    return stax.serial(
        FanOut(2),
        stax.parallel(
            stax.Identity,
            layer
        ),
        FanInSum
    )


def MultiheadSelfAttention(num_heads, output_size, qk_size):
    """
    Multi-head self-attention layer

    Apply function takes in embedded vector and returns self-attention result

    Args:
    num_heads = number of heads of self-attention
    output_size = output dimension, should be same size as the input embedded
                  vector if this layer will be within a residual block
    qk_size = The dimension of the query-key matrix subspace

    Dimension of the value output space must be identical to query-key subspace
    because this is required by jax.nn.dot_product_attention.
    This is not required by the algorithm which can be easily written in terms
    of reshapes and matrix multiplications if separate values are needed.
    """
    vo_size = qk_size  # required by jax.nn.dot_product_attention

    # Define projection layers: query, key, value, output
    q_init, q_apply = Linear_map(qk_size*num_heads)  # Input size: hidden_size
    k_init, k_apply = Linear_map(qk_size*num_heads)  # Input size: hidden_size
    v_init, v_apply = Linear_map(vo_size*num_heads)  # Input size: hidden_size
    o_init, o_apply = Linear_map(output_size)  # Input size: vo_size*num_heads

    def init_fun(rng, input_shape):
        # input_shape: (..., seq_len, embedded_dim)
        rng_q, rng_k, rng_v, rng_o = jax.random.split(rng, 4)

        # Query, key, value, and output parameters
        _,       q_params = q_init(rng_q, input_shape)
        _,       k_params = k_init(rng_k, input_shape)
        shape_v, v_params = v_init(rng_v, input_shape)
        out_shape, o_params = o_init(rng_o, shape_v)
        return out_shape, (q_params, k_params, v_params, o_params)

    def apply_fun(params, inputs, **kwargs):
        # inputs: (..., seq_len, embedded_dim)
        # Last dimension should be output_size if this will be within residual block
        q_params, k_params, v_params, o_params = params

        prior_shape = inputs.shape[:-2]
        seq_len = inputs.shape[-2]
        embedded_dim = inputs.shape[-1]

        # Flatten for Dense projection
        input_flat = inputs.reshape((-1, embedded_dim))
        # projection shapes: (..., seq_len, qk_size*num_heads)
        Q = q_apply(q_params, input_flat)  # (..., seq_len, qk_size*num_heads)
        K = k_apply(k_params, input_flat)  # (..., seq_len, qk_size*num_heads)
        V = v_apply(v_params, input_flat)  # (..., seq_len, vo_size*num_heads)

        Q = Q.reshape((*prior_shape, seq_len, num_heads, qk_size))
        K = K.reshape((*prior_shape, seq_len, num_heads, qk_size))
        V = V.reshape((*prior_shape, seq_len, num_heads, vo_size))

        attended = jax.nn.dot_product_attention(Q, K, V)
        attended = attended.reshape((*prior_shape, seq_len, vo_size*num_heads))

        # Final linear projection
        # (..., seq_len, output_size)
        out = o_apply(o_params, attended)
        return out

    return init_fun, apply_fun


@jit
def leakytanh(x):
    act = 0.975
    return act * jnp.tanh(x) + (1 - act) * x


GELU = elementwise(lambda x: jax.nn.gelu(x, approximate=False))
LeakyTanh = elementwise(leakytanh)
Sin = elementwise(jnp.sin)


class WavefunctionUtil(object):
    def __init__(self, config):
        # Key generation and
        seed = config.seed_net
        key = jax.random.PRNGKey(seed)
        _, key = jax.random.split(key)  # Used to keep same key as previous versions
        self.key = key

        activation_function_dict = {
            'GELU': GELU,
            'Tanh': Tanh,
            'LeakyTanh': LeakyTanh,
            'Sin': Sin
        }

        # Default network structure parameters
        self.activation = activation_function_dict[config.activation_function]
        self.ndense = config.ndense
        self.nlat = config.nlat
        self.tr_perceptrons_per_layer = config.tr_perceptrons_per_layer

    @partial(jit, static_argnums=(0,))
    def remove_cm_util(self, r):
        rcm = jnp.mean(r, axis=0)
        r = (r - rcm[None, :])
        return r

    def dense_net(self, width, depth, out_size):
        """
        Depth includes the output layer so depth-1 is the number of hidden layers

        Constructs the function which creates initial parameters and the
        function which calculates the net. If parallel structures are required
        the depth argument can be input as a tuple/list of integers.
        Otherwise depth should be an integer.
        If using parallel structures then out_size should also be a tuple
        rather than an integer
        """
        parallel = not type(depth) == int

        if not type(depth) == type(out_size):
            raise AssertionError('The inputs depth and out_size must both have array like structure for parallel construction or both be integers otherwise.')

        def net_funcs(depth, out_size):
            layers = []
            for _ in range(depth-1):
                layers += [Dense(width), self.activation]
            layers += [Dense(out_size)]

            return stax.serial(*layers)

        if parallel:
            funcs = [net_funcs(*x) for x in zip(depth, out_size)]

            init_func, apply_func = stax.serial(
                FanOut(len(depth)),
                stax.parallel(*funcs),
                FanInConcat()
                )
        else:
            init_func, apply_func = net_funcs(depth, out_size)

        return init_func, apply_func

    def residual_dense_net(self, width, depth, out_size):
        """Dense(input -> hidden) -> n× ResidualBlock(hidden) -> Dense(hidden->output)."""
        # serial layer to be wrapped by residual
        layer = stax.serial(
            LayerNorm(),
            Dense(width),
            self.activation,
        )

        return stax.serial(
            Linear_map(width),  # Linear layer to get input to 'width' size
            *[residual_block(layer) for _ in range(depth)],
            Dense(out_size),  # Linear layer to map output to 'out_size'
        )

    def init_params(self, init_func, in_shape, count=-1):
        """
        Returns initial parameters given by init_func with given input size.

        in_shape is the required shape for the layer being initialized through
        init_func. If an integer is specified instead of a tuple it is
        converted to a tuple with a -1 prepended.

        If count is specified then a list of these structures is returned
        with length equal to count.

        if count is specified the returned values are in a list even if count is 1.
        """
        assert (type(count) == int) and (count == -1 or count > 0)

        if type(in_shape) == int:
            in_shape = (-1, in_shape)

        key_list = jax.random.split(self.key, num=abs(count)+1)
        self.key = key_list[0]

        def single_params(i):
            _, params = init_func(key_list[i+1], in_shape)
            return params

        if not count == -1:
            params = [single_params(i) for i in range(count)]
        else:
            params = single_params(0)

        return params

    # Transformers
    def transformer_layer(self, num_heads, hidden_size, qk_size):
        # serial layer to be wrapped by residual
        # hidden_size must be the same size as the final axis of the input

        attention_block = stax.serial(
            LayerNorm(),
            MultiheadSelfAttention(num_heads, hidden_size, qk_size),
        )

        perceptron_block = stax.serial(
            LayerNorm(),
            Dense(hidden_size),
            self.activation,
        )
        residual_perceptron = residual_block(perceptron_block)

        transformer = stax.serial(
            residual_block(attention_block),
            *[residual_perceptron for _ in range(self.tr_perceptrons_per_layer)],
        )

        return transformer

    def full_transformer(self, num_heads, hidden_size, qk_size, depth, out_size):
        """
        Embeds the input prior to passing to the first transformer layer and
        uses a final output layer to map to the out_size dimension
        """

        single_transformer = self.transformer_layer(num_heads, hidden_size, qk_size)
        return stax.serial(
            Linear_map(hidden_size),  # Linear embedding layer
            *[single_transformer for _ in range(depth)],
            Dense(out_size),  # Linear layer to map output to 'out_size'
        )

    # Deepsets
    def deepset_build(self, input_size, out_size, num=-1):
        phi_init, self.phi_deepset_apply = self.dense_net(
            self.ndense, 3, self.nlat
            )

        rho_depth = 3
        if type(out_size) == tuple:
            rho_depth = tuple([3]*len(out_size))

        rho_init, self.rho_deepset_apply = self.dense_net(
            self.ndense, rho_depth, out_size
            )

        phi_p_params = self.init_params(
            phi_init, input_size, count=num
            )
        phi_a_params = self.init_params(
            phi_init, input_size, count=num
            )
        rho_p_params = self.init_params(
            rho_init, self.nlat, count=num
            )
        rho_a_params = self.init_params(
            rho_init, self.nlat, count=num
            )

        return [phi_p_params, phi_a_params, rho_p_params, rho_a_params]

    @partial(jit, static_argnums=(0,))
    def deepset_run(self, params, x_i):
        phi_p_params, phi_a_params, rho_p_params, rho_a_params = params

        # Stack parameters if there is more than one deepset
        phi_p_params = self.pytrees_stack(phi_p_params)
        phi_a_params = self.pytrees_stack(phi_a_params)
        rho_p_params = self.pytrees_stack(rho_p_params)
        rho_a_params = self.pytrees_stack(rho_a_params)

        rho_h = vmap(self.single_deepset, in_axes=(0, 0, 0, 0, None))(phi_p_params, phi_a_params, rho_p_params, rho_a_params, x_i)
        return rho_h

    @partial(jit, static_argnums=(0,))
    def single_deepset(self, phi_p_params, phi_a_params, rho_p_params, rho_a_params, x_i):
        phi_p = jax.scipy.special.logsumexp(self.phi_deepset_apply(phi_p_params, x_i), axis=0)
        phi_a = jax.scipy.special.logsumexp(self.phi_deepset_apply(phi_a_params, x_i), axis=0)
        rho_p = self.rho_deepset_apply(rho_p_params, phi_p)
        rho_a = self.rho_deepset_apply(rho_a_params, phi_a)

        return rho_p, rho_a

    @partial(jit, static_argnums=(0,))
    def pytrees_stack(self, pytrees, axis=0):
        results = jax.tree_util.tree_map(lambda *values: jnp.stack(values, axis=axis), *pytrees)
        return results
