import sys, os

import time
import logging
import pickle

# Frameworks:

#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'
#os.environ["XLA_FLAGS"] = '--xla_gpu_deterministic_ops=true'

import jax
import jaxlib

# Setup multi-instance connection. Must be first jax call.
#if "SLURM_JOB_ID" in os.environ:
#    jax.distributed.initialize(local_device_ids=list(range(8)))
#else:
#    jax.distributed.initialize(cluster_detection_method="mpi4py")

# n_local_devices = jax.local_device_count()

jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_disable_jit', True)


import jax.numpy as jnp
from jax import random, grad, jit, vmap, pmap, jacfwd, jacrev
from functools import partial

from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.experimental.multihost_utils import host_local_array_to_global_array
from jax.experimental import mesh_utils

from ConfigParse import parse
from NeuralWavefunction import Wavefunction
from Metropolis import Metropolis
from NuclearPotential import NuclearPotential
from Observables import Observables
from Optimizer import Optimizer


jax.config.update('jax_threefry_partitionable', True)
#jax.config.update('jax_platform_name', 'cpu')#
#cpus = jax.devices("cpu")
#gpus = jax.devices("gpu")

#print ("cpus", cpus )
#print ("gpus", gpus )
#exit()
config_filename = sys.argv[1]
config = parse(config_filename)

n_devices = jax.device_count()
devices = mesh_utils.create_device_mesh((n_devices,))
mesh = Mesh(devices, axis_names=('i',))

# Model save
model_save_path = f"./{config.output_name}.model"

# Set up logging:
logger = logging.getLogger()
# Create a file handler:
hdlr = logging.FileHandler(f'{config.output_name}.log')
# Add formatting to the log:
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
ch = logging.StreamHandler()
logger.addHandler(hdlr)
logger.addHandler(ch)
# Set the default level. Levels here: https://docs.python.org/2/library/logging.html
if jax.process_index() == 0:
    logger.setLevel(logging.DEBUG)
else:
    # Many jit compilation outputs are WARNING level so setting most processes
    # to ERROR level avoids duplicate output except for errors or crashes.
    logger.setLevel(logging.ERROR)

logger.info(f"jax version: {jax.__version__}")
logger.info(f"jaxlib version: {jaxlib.__version__}")

logger.info(f'n_devices = {n_devices}')
logger.info(f'jax device = {jax.devices()}')
logger.info(f'Process index = {jax.process_index()}, jax.local_device_count = {jax.local_device_count()}')

with open(config_filename, 'r') as f:
    logger.info('\nNon-default parameters.\n')
    logger.info(f.read())

with open('DEFAULT.ini', 'r') as f:
    logger.info('\nDefault parameters which are overwritten by main .ini file.\n')
    logger.info(f.read())

# Initialize the network with one batch dimension, ndim, and npart
wavefunction = Wavefunction(config, mesh)

params_init, nparams = wavefunction.build()
logger.info(f"number of parameters = {nparams}")
if nparams > 10*config.nwalk*config.nav:
    logger.warning(f'Many more parameters than samples {nparams} > 10*nwalk*nav')

# Read saved params on file
if (config.module_load):
    with open(model_save_path, 'rb') as file:
        params_read = pickle.load(file)
        params = jax.tree_util.tree_map(wavefunction.update_mix, params_init, params_read)
else:
    params = params_init

# Replicate params over all devices
params = host_local_array_to_global_array(params, mesh, P())

# Initialize Metropolis sampler
metropolis = Metropolis(n_devices, wavefunction, config, logger, mesh)

# Initialize Potential
potential = NuclearPotential(config.pot_name, config.pot_3b_name)

# Initialize Observables
observables = Observables(n_devices, wavefunction, potential, config, mesh)

# Initialize Optimizer
optimizer = Optimizer(n_devices, nparams, wavefunction, observables, config, mesh)

# Metropolis energy calculation
def vmc_run(config, params, it):

    tgen_i = time.time()
    key_o, r_o, sz_o = metropolis.initialize(it, params)
    #Sz, Tz = observables.spin_isospin(sz_o)
    #logger.info(f"initial S_z: {Sz:3f}")
    #logger.info(f"initial T_z: {Tz:3f}")
    tgen_f = time.time()
    logger.info(f"Initial configurations generated, elapsed time: {tgen_f-tgen_i:.3f} seconds")

#    r_test = r_o.reshape((config.nwalk, config.npart, config.ndim))
#    sz_test = sz_o.reshape((config.nwalk, config.npart, 2))
#    energy = observables.energy(params, r_test, sz_test)
#    print('energy =', energy)
#    exit()

    twlk_i = time.time()
    r_stored, sz_stored, acc_r, acc_sz = metropolis.shmap_walk(params, r_o, sz_o, key_o)

#    print('r_stored.shape', r_stored.shape)
#    jnp.set_printoptions(threshold=jnp.inf)
#    print('r_stored', jnp.array(r_stored))



    logger.info(f"Maximum coordinate: {jnp.max(r_stored):.3f}")
    logger.info(f"Minimum coordinate: {jnp.min(r_stored):.3f}")
#    exit()

    twlk_f = time.time()
    logger.info(f"Walk stored, elapsed time: {twlk_f-twlk_i:.3f} seconds")

    tav_i = time.time()

    energy_stored, obs_avg, obs_err = observables.energy_shmap(params, r_stored, sz_stored)
    rho_avg, rho_err, rho_norm = observables.density_shmap(r_stored, sz_stored)
    r2_avg, r2_err, = observables.radius_shmap(r_stored, sz_stored)
    variance = observables.variance_shmap(energy_stored)

    """
    for i in range(config.nav + config.nac):
        # angular_momentum_pmap = observables.angular_momentum_pmap(params, r_stored[:,:,i,:,:], sz_stored[:,:,i,:,:])
        # obs_stored = obs_stored.at[:,:,i,2].set(angular_momentum_pmap[0])
        # obs_stored = obs_stored.at[:,:,i,3:6].set(angular_momentum_pmap[1])
    """

    # energy_stored.block_until_ready()
    tav_f = time.time()
    logger.info(f"Observables computed, elapsed time: {tav_f-tav_i:.3f} seconds")

    if jax.process_index() == 0:
        _ = observables.density_print(rho_avg, rho_err, rho_norm)

    energy = obs_avg[0]
    error = obs_err[0]
    energy_jf = obs_avg[1]
    error_jf = obs_err[1]
    energy_pt = obs_avg[2]
    error_pt = obs_err[2]
    L2 = obs_avg[3]
    error_L2 = obs_err[3]
    Lx = obs_avg[4]
    error_Lx = obs_err[4]
    Ly = obs_avg[5]
    error_Ly = obs_err[5]
    Lz = obs_avg[6]
    error_Lz = obs_err[6]

    delta_p = 0
    if (config.nopt > 1):
        delta_p = optimizer.shmap_optimize(params, r_stored, sz_stored, energy_stored)
    else:
        obs_avg, obs_err = observables.two_body_isospin_shmap(r_stored, sz_stored)

        delta_p = jax.tree_util.tree_map(wavefunction.update_zero, params)

    logger.info(f"process index = {jax.process_index():.3f}")
    logger.info(f"step = {it}, energy = {energy:.3f}, err = {error:.3f}")
    logger.info(f"step = {it}, energy_jf = {energy_jf:.3f}, err = {error_jf:.3f}")
    logger.info(f"step = {it}, energy_pt = {energy_pt:.3f}, err = {error_pt:.3f}")
    logger.info(f"step = {it}, variance = {variance:.3f}")
    logger.info(f"step = {it}, <r^2> = {r2_avg[0]:.3f}, err = {r2_err[0]:.3f}")
    logger.info(f"step = {it}, <r_p^2> = {r2_avg[1]:.3f}, err = {r2_err[1]:.3f}")
    logger.info(f"step = {it}, <r_n^2> = {r2_avg[2]:.3f}, err = {r2_err[2]:.3f}")
    logger.info(f"step = {it}, L^2 = {L2:.3f}, err = {error_L2:.3f}")
    logger.info(f"step = {it}, Lx = {Lx:.3f}, err = {error_Lx:.3f}")
    logger.info(f"step = {it}, Ly = {Ly:.3f}, err = {error_Ly:.3f}")
    logger.info(f"step = {it}, Lz = {Lz:.3f}, err = {error_Lz:.3f}")
    logger.info(f"acceptance coordinates = {acc_r:.3f}")
    logger.info(f"acceptance spin = {acc_sz:.3f}")

    if (not config.nopt > 1):
        logger.info(obs_avg.tolist())
        logger.info(obs_err.tolist())

    return delta_p

# Optimization loop
for it in range(config.nopt):
    ti = time.time()
    delta_p = vmc_run(config, params, it)
    tf = time.time()
    logger.info(f"elapsed time {tf - ti:.3f}")
    logger.info(f"\n")

    # ALE BACK TO UNFLATTEN  params = params + delta_p

    params = jax.tree_util.tree_map(wavefunction.update_add, params, delta_p)


    # Write saved params on file
    if (jax.process_index() == 0 and config.module_save and config.nopt > 1):
        with open(model_save_path, 'wb') as file:
            pickle.dump(params, file)
