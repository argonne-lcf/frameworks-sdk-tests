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
if "SLURM_JOB_ID" in os.environ:
    jax.distributed.initialize(local_device_ids=list(range(8)))
else:
    jax.distributed.initialize(cluster_detection_method="mpi4py")

# n_local_devices = jax.local_device_count()

jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_disable_jit', True)

import numpy as np
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
from NeuraLIT import WavefunctionDipole, WavefunctionLIT, NeuraLIT


jax.config.update('jax_threefry_partitionable', True)
#jax.config.update('jax_platform_name', 'cpu')#
#cpus = jax.devices("cpu")
#gpus = jax.devices("gpu")

#print ("cpus", cpus )
#print ("gpus", gpus )
#exit()
config_filename = sys.argv[1]
config = parse(config_filename)
config_lit = parse(config_filename)
#config_lit.amplitude_limit = 10
#config_lit.conf = 0.2

n_devices = jax.device_count()
devices = mesh_utils.create_device_mesh((n_devices,))
mesh = Mesh(devices, axis_names=('i',))

# Model save
model_save_path = f"./{config.output_name}.model"
model_lit_save_path = f"./{config.output_name}.model.lit"

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

# Initialize the network for the ground-state, the dipole, and lit

wavefunction_gs = Wavefunction(config, mesh, parity=config.parity)
params_gs, nparams = wavefunction_gs.build()

logger.info(f"number of parameters = {nparams}")
if nparams > 10*config.nwalk*config.nav:
    logger.warning(f'Many more parameters than samples {nparams} > 10*nwalk*nav')

# Read saved params on file for the ground-state wave function 
if (config.module_load):
    with open(model_save_path, 'rb') as file:
        params_gs = pickle.load(file)
else:
    logger.error(f'A LIT calculation requires loading the ground-state wave function parameters. ')

wavefunction_lit_aux = Wavefunction(config_lit, mesh, parity=-config_lit.parity)
params_lit, nparams_lit = wavefunction_lit_aux.build()

# Read saved params on file for the LIT
if (config.module_lit_load):
    with open(model_lit_save_path, 'rb') as file:
        params_lit_read = pickle.load(file)
        params_lit = jax.tree_util.tree_map(wavefunction_lit_aux.update_mix, params_lit, params_lit_read)

wavefunction_dipole = WavefunctionDipole(wavefunction_gs, mesh)
wavefunction_lit = WavefunctionLIT(wavefunction_lit_aux, wavefunction_dipole, mesh, params_gs)
#params_lit = params_lit + [0.5]
#nparams_lit = nparams_lit + 1

# Initialize Potential
potential = NuclearPotential(config.pot_name, config.pot_3b_name)

# Initialize Observables relevant for the ground-state and LIT calculation
observables_gs = Observables(n_devices, wavefunction_gs, potential, config, mesh)
observables_lit = Observables(n_devices, wavefunction_lit, potential, config_lit, mesh)

# Replicate ground-state and lit params over all devices
params_gs = host_local_array_to_global_array(params_gs, mesh, P())
params_lit = host_local_array_to_global_array(params_lit, mesh, P())

# Initialize Metropolis sampler from the ground-state and dipole wave function
metropolis_gs = Metropolis(n_devices, wavefunction_gs, config, logger, mesh)
metropolis_dipole = Metropolis(n_devices, wavefunction_dipole, config, logger, mesh)

config_lit.nav = 1
config_lit.nac = 0
config_lit.neq = 2
#metropolis_lit = Metropolis(n_devices, wavefunction_lit, config_lit, logger, mesh)
#metropolis_hlit = Metropolis(n_devices, wavefunction_hlit, config_lit, logger, mesh)



# Ground-state energy calculation
def ground_state_run(it, params_gs):

    tgen_i = time.time()
    key_o, r_o, sz_o = metropolis_gs.initialize(it, params_gs)
    tgen_f = time.time()
    logger.info(f"Initial configurations generated from ground-state, elapsed time: {tgen_f-tgen_i:.3f} seconds")

    twlk_i = time.time()
    r_stored, sz_stored, acc_r, acc_sz = metropolis_gs.shmap_walk(params_gs, r_o, sz_o, key_o)
    twlk_f = time.time()
    logger.info(f"Walk from ground-state stored, elapsed time: {twlk_f-twlk_i:.3f} seconds")

    tav_i = time.time()
    energy_stored, obs_avg, obs_err = observables_gs.energy_shmap(params_gs, r_stored, sz_stored)
    r2_avg, r2_err, = observables_gs.radius_shmap(r_stored, sz_stored)

    energy = obs_avg[0]
    error = obs_err[0]
    energy_jf = obs_avg[1]
    error_jf = obs_err[1]
    logger.info(f"process index = {jax.process_index():.3f}")
    logger.info(f"step = {it}, energy = {energy:.3f}, err = {error:.3f}")
    logger.info(f"step = {it}, energy_jf = {energy_jf:.3f}, err = {error_jf:.3f}")
    logger.info(f"step = {it}, <r^2> = {r2_avg[0]:.3f}, err = {r2_err[0]:.3f}")
    logger.info(f"step = {it}, <r_p^2> = {r2_avg[1]:.3f}, err = {r2_err[1]:.3f}")
    logger.info(f"step = {it}, <r_n^2> = {r2_avg[2]:.3f}, err = {r2_err[2]:.3f}")
    logger.info(f"acceptance coordinates = {acc_r:.3f}")
    logger.info(f"acceptance spin / isospin = {acc_sz:.3f}")

    return energy

#energy_gs = ground_state_run(0, params_gs)

#energy_gs = -2.242
energy_gs = -28.2
#energy_gs = -37.9
#energy_gs = -87.4

print('ground-state energy', energy_gs)
print('nparams_lit', nparams_lit)

# Initialize LIT
neuralit = NeuraLIT(wavefunction_dipole, wavefunction_lit, observables_lit, energy_gs, mesh, config, nparams_lit)

twlk_i = time.time()
key_d, r_d, sz_d = metropolis_dipole.initialize(0, params_gs)
r_d, sz_d, acc_r_d, acc_sz_d = metropolis_dipole.shmap_walk(params_gs, r_d, sz_d, key_d)
print('r_d.shape', r_d.shape)
print('sz_d.shape', sz_d.shape)

twlk_f = time.time()
logger.info(f"Walk from dipole stored, elapsed time: {twlk_f-twlk_i:.3f} seconds")
logger.info(f"acceptance coordinates dipole = {acc_r_d:.3f}")
logger.info(f"acceptance spin/isospin dipole = {acc_sz_d:.3f}")

sig_I = 10

# Compute the dipole sum rule, as well as the ground-state and dipole wave functions for all averaging blocks
def sum_rule_compute(r_d, sz_d, params_gs):
    sum_rule_list = []
    logpsi_dipole_list = []
    logpsi_gs_list = []
    for i in range(config.nav):
        r_d_batch = r_d[:,i, :, :]
        sz_d_batch = sz_d[:,i, :, :]
        logpsi_dipole = wavefunction_dipole.shard_map_logpsi(params_gs, r_d_batch, sz_d_batch)
        logpsi_dipole_list.append(logpsi_dipole)
        logpsi_gs = wavefunction_gs.shard_map_logpsi(params_gs, r_d_batch, sz_d_batch)
        logpsi_gs_list.append(logpsi_gs)
        sum_rule = jnp.mean(jnp.abs(jnp.exp(2 * (logpsi_gs - logpsi_dipole))))
        sum_rule_list.append(sum_rule)
    return jnp.asarray(logpsi_dipole_list), jnp.asarray(logpsi_gs_list), jnp.asarray(sum_rule_list)

# Solves the LIT equation (H - E_0 - sigma_R - i sigma_I) |psi_L> = D |psi_0> using Adam for now
def LIT_optimize(r_d, sz_d, params_lit, d_overlap, m_overlap, g2_overlap, sig_R, sig_I, key_batch):
    overlap_list = []
    for it in range(config.nopt):
        #ibatch = jnp.mod(it, config.nav)
        #r_d_batch = r_d[:,ibatch, :, :]
        #sz_d_batch = sz_d[:,ibatch, :, :]
        #r_l = r_d_batch
        #sz_l = sz_d_batch

        key_batch, key_input = jax.random.split(key_batch)
        ibatch_d = jax.random.randint(key_input, shape=(), minval=0, maxval=config.nav)
        key_batch, key_input = jax.random.split(key_batch)
        ibatch_l = jax.random.randint(key_input, shape=(), minval=0, maxval=config.nav)

        logger.info(f'ibatch_d, {ibatch_d}')
        logger.info(f'ibatch_l, {ibatch_l}')
        r_d_batch = r_d[:,ibatch_d, :, :]
        sz_d_batch = sz_d[:,ibatch_d, :, :]
        r_l_batch = r_d[:,ibatch_l, :, :]
        sz_l_batch = sz_d[:,ibatch_l, :, :]


#        key_l, r_l, sz_l = metropolis_hlit.initialize(it+1, params_lit)
#        params_hlit = params_lit + [sig_R] + [sig_I] + [energy_gs]
#        print('energy_gs', params_hlit[-1])
#        print('sig_I', params_hlit[-2])
#        print('sig_R', params_hlit[-3])
#        r_l, sz_l, acc_r_l, acc_sz_l = metropolis_hlit.shmap_walk(params_hlit, r_l, sz_l, key_l)
#        logger.info(f"acceptance coordinates lit = {acc_r_l:.3f}")
#        logger.info(f"acceptance spin/isospin lit = {acc_sz_l:.3f}")
#        r_l_batch = r_l[:, config_lit.nav, :, :]
#        sz_l_batch = sz_l[:, config_lit.nav, :, :]

        r_l_batch = r_d_batch
        sz_l_batch = sz_d_batch
#        overlap, overlap_lhs, overlap_rhs, overlap_test, d_overlap, m_overlap, g2_overlap = \
#        neuralit.shard_map_adam_overlap(it, params_lit, params_gs, r_l_batch, sz_l_batch, r_d_batch, sz_d_batch, m_overlap, g2_overlap, sig_R, sig_I)
        overlap, overlap_lhs, overlap_rhs, overlap_test, d_overlap, m_overlap, g2_overlap = \
        neuralit.shard_map_sr_overlap(it, params_lit, params_gs, r_l_batch, sz_l_batch, r_d_batch, sz_d_batch, m_overlap, g2_overlap, sig_R, sig_I )
#        overlap, overlap_lhs, overlap_rhs, overlap_test, d_overlap, m_overlap, g2_overlap = \
#        neuralit.shard_map_sr_filippo(it, params_lit, params_gs, r_l_batch, sz_l_batch, r_d_batch, sz_d_batch, m_overlap, g2_overlap, sig_R, sig_I )
        overlap_list.append(overlap)
        n_avg = 20
        overlap_last = overlap_list[-n_avg:]
        overlap_average = sum(overlap_last) / len(overlap_last)

        logger.info(f'overlap {it}, {overlap}')
        logger.info(f'overlap lhs {it}, {overlap_lhs}')
        logger.info(f'overlap rhs {it}, {overlap_rhs}')
        logger.info(f'overlap test {it}, {overlap_test}')
        #if it == config.nopt - 1:
        logger.info(f'overlap average {it}, {overlap_average}')

        params_lit = jax.tree_util.tree_map(wavefunction_lit.update_add, params_lit, d_overlap)
    # Skip to the next iteration if |overlap - 1| < 0.0001
        if abs(overlap_average - 1) < 0.00001:
         #  logger.info(f'overlap average {it}, {overlap_average}')
           logger.info(f'Exiting loop at iteration {it} as |overlap - 1| < 0.0001')
           break
    return params_lit, d_overlap, m_overlap, g2_overlap, key_batch, overlap_average

def LIT_estimate(r_d, sz_d, params_lit, logpsi_dipole, logpsi_gs, sig_R, sig_I, sum_rule):
    lit_xilin_list = []
    error_bound_list = []
    for i in range(config.nav):
        r_d_batch = r_d[:,i, :, :]
        sz_d_batch = sz_d[:,i, :, :]
        logpsi_dipole_batch = logpsi_dipole[i]
        logpsi_gs_batch = logpsi_gs[i]
        lit_xilin, error_bound = neuralit.lit_compute_xilin(params_lit, r_d_batch, sz_d_batch, logpsi_dipole_batch, logpsi_gs_batch, sig_R, sig_I, sum_rule)
        lit_xilin_list.append(lit_xilin)
        error_bound_list.append(error_bound)
    return jnp.asarray(lit_xilin_list), jnp.asarray(error_bound_list)


logpsi_dipole, logpsi_gs, sum_rule = sum_rule_compute(r_d, sz_d, params_gs)
sum_rule_avg = 1 / jnp.average(sum_rule)
sum_rule_std = jnp.std(sum_rule)
sum_rule_err = sum_rule_std / jnp.sqrt(sum_rule.size)
print('sum_rule =', sum_rule_avg, '+-', sum_rule_avg**2 * sum_rule_err)

n_sig_R = 51
sig_R_list = jnp.linspace(0, 100, n_sig_R)

d_overlap = jnp.zeros(nparams)
m_overlap = jnp.zeros(nparams)
g2_overlap = jnp.zeros(nparams)

sig_R = sig_R_list[0] 
params_hlit = params_lit + [sig_R] + [sig_I] + [energy_gs]
print('energy_gs outside', params_hlit[-1])
print('sig_I outside', params_hlit[-2])
print('sig_R outside', params_hlit[-3])

key_batch = random.PRNGKey(config.seed_walk)
#key_l, r_l, sz_l = metropolis_hlit.initialize(0, params_hlit)
#r_l, sz_l, acc_r_l, acc_sz_l = metropolis_hlit.shmap_walk(params_hlit, r_l, sz_l, key_l)
for i in range(n_sig_R):
    sig_R = sig_R_list[i] 
    if i == 1:
       config.nopt = 400
    params_lit, d_overlap, m_overlap, g2_overlap, key_batch, overlap_average = LIT_optimize(r_d, sz_d, params_lit, d_overlap, m_overlap, g2_overlap, sig_R, sig_I, key_batch)
    lit_xilin, error_bound = LIT_estimate(r_d, sz_d, params_lit, logpsi_dipole, logpsi_gs, sig_R, sig_I, sum_rule)

  #  fname = 'lit_xilin.dat'
  #  np.savetxt(fname, lit_xilin)
  #  print('lit xilin all [', ', '.join(f'{x:.6e}' for x in lit_xilin), ']')

    lit_avg = jnp.average(lit_xilin)
    lit_std = jnp.std(lit_xilin)
    lit_err = lit_std / jnp.sqrt(lit_xilin.size)

    err_avg = jnp.average(error_bound)
    err_max = jnp.max(error_bound)

    logger.info(f'average = {sig_R:15.7e}, {lit_avg:15.7e}, {lit_err:15.7e}, {err_avg:15.7e}, {overlap_average:15.7e}') 
  #  print(f'average = {sig_R:15.7e}, {lit_avg:15.7e} ± {lit_err:15.7e}')

    
#    bound_1, bound_2 = neuralit.lit_error_bound(params_lit, r_d[:,0, :, :], sz_d[:,0, :, :], logpsi_dipole[0], sig_R, sig_I, sum_rule_avg, lit_avg)
  
#    logger.info(f'average = {sig_R:15.7e}, {lit_avg:15.7e} ± {lit_err:15.7e} ± {bound_1:15.7e} ± {bound_2:15.7e} ') 


# Write saved params on file
if (config.module_lit_save):
    with open(model_lit_save_path, 'wb') as file:
        pickle.dump(params_lit, file)
