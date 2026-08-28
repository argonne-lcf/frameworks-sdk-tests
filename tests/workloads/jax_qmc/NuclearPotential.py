import sys, os
import numpy as np
import jax.numpy as jnp
from jax import random, grad, jit, vmap, jacfwd, jacrev
from functools import partial


class NuclearPotential(object):
    def __init__(self, pot_name, pot_3b_name):
        self.hc = 197.327053
        self.alpha = 1 / 137.03599
        self.M_p = 938.27231
        self.M_n = 939.56563
        self.mu_p = 2.79285
        self.mu_n = -1.91304

        self.beta_n = 0.0189  # fm^2
        self.b = 4.27  # fm^-1

        self.pot_3b_type = 'Linear'

        # Set up two body
        self.v_em = self.v_em_coul
        if (pot_name == 'pionless_2'):
            self.vkr = 2.0
            self.v0r = -133.3431
            self.v0s = -9.0212
            self.ar3b = np.sqrt(68.48830)
            self.v_2b = self.pionless_2b
            self.v_3b = self.pionless_3b
        elif (pot_name == 'pionless_4'):
            self.vkr = 4.0
            self.v0r = -487.6128
            self.v0s = -17.5515
            self.ar3b = np.sqrt(677.79890)
            self.v_2b = self.pionless_2b
            self.v_3b = self.pionless_3b
        elif (pot_name == 'pionless_6'):
            self.vkr = 6.0
            self.v0r = -1064.5010
            self.v0s = -26.0830
            self.ar3b = np.sqrt(2652.65100)
            self.v_2b = self.pionless_2b
            self.v_3b = self.pionless_3b
        elif (pot_name == 'pionless_lo_a'):
            self.R0 = 1.7
            self.R1 = 1.5
            self.C01 = -4.38524414
            self.C10 = -8.00783936
            lamchi = 1000./self.hc
            fpi = 92.40/self.hc
            if (pot_3b_name == 'R3_1.0'):
                self.R3 = 1.0
                self.ce3b = 1.8354
            elif (pot_3b_name == 'R3_1.5'):
                self.R3 = 1.5
                self.ce3b = 4.6301
            elif (pot_3b_name == 'R3_2.0'):
                self.R3 = 2.0
                self.ce3b = 11.6871
            elif (pot_3b_name == 'R3_2.5'):
                self.R3 = 2.5
                self.ce3b = 27.4702
            else:
                self.R3 = 1.
                self.ce3b = 0
            self.ce3b = jnp.sqrt(self.ce3b / lamchi / fpi**4 * self.hc / jnp.pi**3 / self.R3**6)
            self.v_2b = self.pionless_2b_lo
            self.v_3b = self.pionless_3b_lo
        elif (pot_name == 'pionless_lo_o'):
            self.R0 = 1.54592984
            self.R1 = 1.83039397
            self.C01 = -5.27518671
            self.C10 = -7.04040080
            lamchi = 1000./self.hc
            fpi = 92.40/self.hc
            if (pot_3b_name == 'R3_1.0'):
                self.R3 = 1.0
                self.ce3b = 1.0786
            elif (pot_3b_name == 'R3_1.1'):
                self.R3 = 1.1
                self.ce3b = 1.2945
            elif (pot_3b_name == 'R3_1.5'):
                self.R3 = 1.5
                self.ce3b = 2.7676
            elif (pot_3b_name == 'R3_2.0'):
                self.R3 = 2.0
                self.ce3b = 6.95356
            elif (pot_3b_name == 'R3_2.5'):
                self.R3 = 2.5
                self.ce3b = 16.21993
            else:
                self.R3 = 1.
                self.ce3b = 0
            self.ce3b = jnp.sqrt(self.ce3b / lamchi / fpi**4 * self.hc / jnp.pi**3 / self.R3**6)
            self.v_2b = self.pionless_2b_lo
            self.v_3b = self.pionless_3b_lo
        elif (pot_name == 'p_wave'):
            # fm
            self.R01 = 1.8310
            self.R10 = 1.5579
            self.R00 = 4.03
            self.R11 = 3.35

            # MeV
            self.V01 = -30.545885
            self.V10 = -66.5824776
            self.V00 = 1.625
            # self.V11 = None  # set below for fitting

            self.pot_3b_type = 'Linear_with_projector'
            if (pot_3b_name == 'weak'):
                self.V11 = -1.78
                Z0 = 4.265
                self.R3 = 1.733
            elif (pot_3b_name == 'strong'):
                self.V11 = -3.857
                Z0 = 2.6
                self.R3 = 2.175
            else:
                print(f'Three body name "{pot_3b_name}" not defined for {pot_name} potential.')
                raise ValueError(f'Three body name "{pot_3b_name}" not defined for {pot_name} potential.')
                quit()

            # There Z_0 -> sqrt(Z_0) for the coefficient of t_ij
            self.ce3b = jnp.sqrt(Z0)

            self.v_2b = self.pionless_2b_lo_pwave
            self.v_3b = self.pionless_3b_lo
        elif (pot_name == 'p_wave_double_exp'):
            # fm
            self.R01 = 1.8310
            self.R10 = 1.5579
            self.R00_a = 3.68008
            #self.R11_a = 1.13687
            self.R11_a = 4.94292  # average of J=0,1,2
            self.R00_b = 2.38968
            #self.R11_b = 3.11128
            self.R11_b = 2.91940  # average of J=0,1,2

            # MeV
            self.V01 = -30.545885
            self.V10 = -66.5824776
            self.V00_a = 2.49875
            #self.V11_a = 0.60232
            self.V11_a = 0.44996  # average of J=0,1,2
            self.V00_b = -1.52150
            #self.V11_b = -4.47425
            self.V11_b = -4.22578  # average of J=0,1,2

            # Three body fit not changed from original p-wave
            self.pot_3b_type = 'Linear_with_projector'
            if (pot_3b_name == 'weak'):
                Z0 = 4.265
                self.R3 = 1.733
            elif (pot_3b_name == 'strong'):
                Z0 = 2.6
                self.R3 = 2.175
            else:
                print(f'Three body name "{pot_3b_name}" not defined for {pot_name} potential.')
                raise ValueError(f'Three body name "{pot_3b_name}" not defined for {pot_name} potential.')
                quit()

            # There Z_0 -> sqrt(Z_0) for the coefficient of t_ij
            self.ce3b = jnp.sqrt(Z0)

            self.v_2b = self.pionless_2b_lo_pwave_double_exp
            self.v_3b = self.pionless_3b_lo
        elif (pot_name == 'pionless_Anthony'):
            self.R0 = 1.53667  # 3S1 regulator cutoff
            self.R1 = 1.81287  # 1S0 regulator cutoff
            self.C01 = -5.15525  # 1S0 LO LEC
            self.C10 = -6.97622  # 3S1 LO LEC
            self.C0_IT = 0.01860  # isotensor term LEC
            self.C_CA = 0.00830  # \tau_z term LEC

            lamchi = 1000./self.hc
            fpi = 92.40/self.hc
            # 3 body fit to 16O
            if (pot_3b_name == 'R3_1.4'):
                self.R3 = 1.4
                self.pot_3b_type = 'Linear_with_projector'
                self.ce3b = 2.012
            elif (pot_3b_name == 'R3_1.9'):
                self.R3 = 1.9
                self.pot_3b_type = 'Triangle_with_projector'
                self.ce3b = 11.969
            elif (pot_3b_name == 'None'):
                self.R3 = 1.
                self.ce3b = 0
            else:
                raise ValueError('No defined three body type selected.')
            self.v_2b = self.pionless_2b_CI_CD_CA
            self.v_em = self.v_em_ext
            self.ce3b = jnp.sqrt(self.ce3b / lamchi / fpi**4 * self.hc / jnp.pi**3 / self.R3**6)
            self.v_3b = self.pionless_3b_lo

        elif (pot_name == 'pionless_contessi_100'):
            self.Cs=-24.865368
            self.Ct=-45.827736
            self.Rs=1.00
            self.Rt=1.00
            self.R3=1.00
            self.D0=0.941365
            self.sign_3b=1
            self.v_2b = self.pionless_2b_contessi
            self.v_3b = self.pionless_3b_contessi
            self.v_em = self.v_em_off
        elif (pot_name == 'pionless_contessi_125'):
            self.Cs=-39.701746
            self.Ct=-63.859202
            self.Rs=0.8
            self.Rt=0.8
            self.R3=0.8
            self.D0=7.293291
            self.sign_3b=1
            self.v_2b = self.pionless_2b_contessi
            self.v_3b = self.pionless_3b_contessi
            self.v_em = self.v_em_off
        elif (pot_name == 'pionless_contessi_150'):
            self.Cs=-58.014561
            self.Ct=-85.555297
            self.Rs=0.66666666
            self.Rt=0.66666666
            self.R3=0.66666666
            self.D0=18.389272
            self.sign_3b=1
            self.v_2b = self.pionless_2b_contessi
            self.v_3b = self.pionless_3b_contessi
            self.v_em = self.v_em_off
        elif (pot_name == 'pionless_contessi_175'):
            self.Cs=-79.804620
            self.Ct=-110.83083
            self.Rs=0.57142857142
            self.Rt=0.57142857142
            self.R3=0.57142857142
            self.D0=35.001816
            self.sign_3b=1
            self.v_2b = self.pionless_2b_contessi
            self.v_3b = self.pionless_3b_contessi
            self.v_em = self.v_em_off

        elif (pot_name == 'anlv4prime'):
            fname = 'potential/av4p/pot_nn'
            data = np.loadtxt(fname, unpack=True, skiprows = 1)
            r_tab = data[0,:]
            v_tab = data[1:7,:]
            print('r_tab=', r_tab)
            print('v_tab')
            ntab = r_tab.shape[0]
            for i in range (ntab):
                print("v_tab", v_tab[0,i], v_tab[1,i], v_tab[2,i], v_tab[3,i], v_tab[4,i], v_tab[5,i])

            self.r_tab = jnp.array(r_tab)
            self.v_tab = jnp.array(v_tab)
            self.v_2b = self.argonne
            self.v_3b = self.uix
            self.v_em = self.v_em_off

        elif (pot_name == 'anlv6prime'):
            fname = 'potential/av6p/pot_nn'
            data = np.loadtxt(fname, unpack=True, skiprows = 1)
            r_tab = data[0,:]
            v_tab = data[1:7,:]
            print('r_tab=', r_tab)
            print('v_tab')
            ntab = r_tab.shape[0]
            for i in range (ntab):
                print("v_tab", v_tab[0,i], v_tab[1,i], v_tab[2,i], v_tab[3,i], v_tab[4,i], v_tab[5,i])

            self.r_tab = jnp.array(r_tab)
            self.v_tab = jnp.array(v_tab)
            self.v_2b = self.argonne
            self.v_3b = self.uix
            self.v_em = self.v_em_coul

        nr_test = 100
        r_test = jnp.linspace(0, 5, nr_test)
        pot_test = np.zeros(shape=(1+8, nr_test))  # r, v_2b, v_em
        pot_em_pp_test = np.zeros(shape=(1+8, nr_test))  # r, v_2b, v_em
        pot_em_np_test = np.zeros(shape=(1+8, nr_test))  # r, v_2b, v_em
        pot_em_nn_test = np.zeros(shape=(1+8, nr_test))  # r, v_2b, v_em
        pot_test[0, :] = r_test
        pot_em_pp_test[0,:] = r_test
        pot_em_np_test[0,:] = r_test
        pot_em_nn_test[0,:] = r_test
        for i in range(r_test.shape[0]):
            pot_test[1:9, i] = self.v_2b(r_test[i])
            em_temp = self.v_em(r_test[i])
            pot_em_pp_test[1:9, i] = em_temp[0]
            pot_em_np_test[1:9, i] = em_temp[1]
            pot_em_nn_test[1:9, i] = em_temp[2]
        np.savetxt("pot_nn.dat", np.transpose(pot_test), fmt='%15.7e', delimiter=' ', newline=os.linesep)
        np.savetxt("pot_em_pp.dat", np.transpose(pot_em_pp_test), fmt='%15.7e', delimiter=' ', newline=os.linesep)
        np.savetxt("pot_em_np.dat", np.transpose(pot_em_np_test), fmt='%15.7e', delimiter=' ', newline=os.linesep)
        np.savetxt("pot_em_nn.dat", np.transpose(pot_em_nn_test), fmt='%15.7e', delimiter=' ', newline=os.linesep)
        

    @partial(jit, static_argnums=(0,))
    def pionless_2b(self, rr):
        pot_2b = jnp.zeros(8)
        x = self.vkr * rr
        vr = jnp.exp(-x**2 / 4.0)
        pot_2b = pot_2b.at[0].set(self.v0r * vr)
        pot_2b = pot_2b.at[2].set(self.v0s * vr)
        return pot_2b

    @partial(jit, static_argnums=(0,))
    def pionless_3b(self, rr):
        x = self.vkr * rr
        vr = jnp.exp(-x**2 / 4.0)
        pot_3b = self.ar3b * vr
        return pot_3b

    @partial(jit, static_argnums=(0,))
    def pionless_2b_lo(self, rr):
        pot_2b = jnp.zeros(8)
        C0_r = 1. / (jnp.sqrt(jnp.pi)*self.R0)**3*jnp.exp( -( rr / self.R0 )**2 )
        C1_r = 1. / (jnp.sqrt(jnp.pi)*self.R1)**3*jnp.exp( -( rr / self.R1 )**2 )
        pot_2b = pot_2b.at[0].set( 3. * ( self.C01 * C1_r + self.C10 * C0_r ) )
        pot_2b = pot_2b.at[1].set( self.C01 * C1_r - 3. * self.C10 * C0_r )
        pot_2b = pot_2b.at[2].set( -3. * self.C01 * C1_r + self.C10 * C0_r )
        pot_2b = pot_2b.at[3].set( -1. * ( self.C01 * C1_r + self.C10 * C0_r ) )
        pot_2b = pot_2b / 16. * self.hc
        return pot_2b

    @partial(jit, static_argnums=(0,))
    def pionless_2b_lo_pwave(self, rr):
        pot_2b = jnp.zeros(8)
        # To simplify, I'm going to multiply VST by CST since this always
        # happens for this potential
        # So the variable name isn't accurate for the CST_r
        C01_r = self.V01*jnp.exp(-(rr / self.R01)**2)
        C10_r = self.V10*jnp.exp(-(rr / self.R10)**2)
        C00_r = self.V00*jnp.exp(-(rr / self.R00)**2)
        C11_r = self.V11*jnp.exp(-(rr / self.R11)**2)

        pot_2b = pot_2b.at[0].set( 3.*C01_r + 3.*C10_r + 1.*C00_r + 9.*C11_r)
        pot_2b = pot_2b.at[1].set( 1.*C01_r - 3.*C10_r - 1.*C00_r + 3.*C11_r)
        pot_2b = pot_2b.at[2].set(-3.*C01_r + 1.*C10_r - 1.*C00_r + 3.*C11_r)
        pot_2b = pot_2b.at[3].set(-1.*C01_r - 1.*C10_r + 1.*C00_r + 1.*C11_r)
        pot_2b = pot_2b / 16.
        return pot_2b

    @partial(jit, static_argnums=(0,))
    def pionless_2b_lo_pwave_double_exp(self, rr):
        pot_2b = jnp.zeros(8)
        # To simplify, I'm going to multiply VST by CST since this always
        # happens for this potential
        # So the variable name isn't accurate for the CST_r
        C01_r = self.V01*jnp.exp(-(rr / self.R01)**2)
        C10_r = self.V10*jnp.exp(-(rr / self.R10)**2)
        C00_r = self.V00_a*jnp.exp(-(rr / self.R00_a)**2) + self.V00_b*jnp.exp(-(rr / self.R00_b)**2)
        C11_r = self.V11_a*jnp.exp(-(rr / self.R11_a)**2) + self.V11_b*jnp.exp(-(rr / self.R11_b)**2)

        pot_2b = pot_2b.at[0].set( 3.*C01_r + 3.*C10_r + 1.*C00_r + 9.*C11_r)
        pot_2b = pot_2b.at[1].set( 1.*C01_r - 3.*C10_r - 1.*C00_r + 3.*C11_r)
        pot_2b = pot_2b.at[2].set(-3.*C01_r + 1.*C10_r - 1.*C00_r + 3.*C11_r)
        pot_2b = pot_2b.at[3].set(-1.*C01_r - 1.*C10_r + 1.*C00_r + 1.*C11_r)
        pot_2b = pot_2b / 16.
        return pot_2b

    @partial(jit, static_argnums=(0,))
    def pionless_2b_CI_CD_CA(self, rr):
        pot_2b = jnp.zeros(8)
        C0_r = 1. / (jnp.sqrt(jnp.pi)*self.R0)**3*jnp.exp(-(rr / self.R0)**2)
        C1_r = 1. / (jnp.sqrt(jnp.pi)*self.R1)**3*jnp.exp(-(rr / self.R1)**2)

        # CI terms
        CI_0 = 3. * (self.C01 * C1_r + self.C10 * C0_r)/16.0
        CI_1 = (self.C01 * C1_r - 3. * self.C10 * C0_r)/16.0
        CI_2 = (-3. * self.C01 * C1_r + self.C10 * C0_r)/16.0
        CI_3 = -1. * (self.C01 * C1_r + self.C10 * C0_r)/16.0

        # CD terms; T12 = 3*tau_iz*tau_jz- DOTPRODUCT(tau_i, tau_j)
        CD_4 = self.C0_IT*C1_r

        # CA term; tau_iz + tau_jz
        CA_5 = self.C_CA*C1_r

        pot_2b = pot_2b.at[0].set(CI_0)
        pot_2b = pot_2b.at[1].set(CI_1)
        pot_2b = pot_2b.at[2].set(CI_2)
        pot_2b = pot_2b.at[3].set(CI_3)
        pot_2b = pot_2b.at[6].set(CD_4)
        pot_2b = pot_2b.at[7].set(CA_5)
        pot_2b = pot_2b * self.hc
        return pot_2b

    @partial(jit, static_argnums=(0,))
    def pionless_3b_lo(self, rr):
        pot_3b = self.ce3b * jnp.exp(- (rr / self.R3)**2)
        return pot_3b

    @partial(jit, static_argnums=(0,))
    def v_em_coul(self, rr):
        rr = jnp.maximum(rr,0.0001)
        br = self.b * rr
        fcoul = 1 - (1 + 11 * br / 16 + 3 * br**2 / 16 + br**3 / 48) * jnp.exp(-br)
        pot_em = self.alpha * self.hc * fcoul / rr
        pot_em_pp = jnp.zeros(8)
        pot_em_np = jnp.zeros(8)
        pot_em_nn = jnp.zeros(8)
        pot_em_pp = pot_em_pp.at[0].set(pot_em)
        return pot_em_pp, pot_em_np, pot_em_nn

    @partial(jit, static_argnums=(0,))
    def v_em_off(self, rr):
        pot_em_pp = jnp.zeros(8)
        pot_em_np = jnp.zeros(8)
        pot_em_nn = jnp.zeros(8)
        return pot_em_pp, pot_em_np, pot_em_nn

    @partial(jit, static_argnums=(0,))
    def v_em_ext(self, rr):
        # Define isospin projectors (tz values are 1 or -1)
        pot_em_pp = jnp.zeros(8)
        pot_em_np = jnp.zeros(8)
        pot_em_nn = jnp.zeros(8)

        rr = jnp.maximum(rr, 0.0001)
        br = self.b * rr  # Defined as x, unitless

        # proton-proton terms
        fcoul = 1 - (1 + 11 * br / 16. + 3 * br**2 / 16. + br**3 / 48.) * jnp.exp(-br)  # Unitless
        fdelta = self.b**3*(1 / 16. + br / 16. + br**2 / 48.)*jnp.exp(-br)  # fm^-3
        gamma = 0.577216
        M_e = 0.51099895
        kr = M_e * rr / self.hc
        vp_integral = -gamma - 5.0/6.0 + jnp.abs(jnp.log(kr)) + 6.0*jnp.pi*kr/8.0

        VC1_pp = (self.alpha * fcoul / rr) * self.hc
        VC2 = (-1*(self.alpha*fcoul/rr)**2/self.M_p) * self.hc**2
        VDF = (-self.alpha/(4*self.M_p**2)*fdelta) * self.hc**3
        VVP = ((2*self.alpha**2*fcoul)/(3*jnp.pi*rr)*vp_integral) * self.hc
        VMM_pp = (-(2*self.alpha*self.mu_p**2)/(3*4*self.M_p**2)*fdelta) * self.hc**3

        pot_em_pp = pot_em_pp.at[0].set(VC1_pp+VC2+VDF+VVP)
        pot_em_pp = pot_em_pp.at[2].set(VMM_pp)

        # neutron-proton terms
        fnp = self.b**2*(15*br+15*br**2+6*br**3+br**4)*jnp.exp(-br)/384.0  # fm^-2

        VC1_np = (self.alpha*self.beta_n*fnp/rr) * self.hc
        VMM_np = (-(2*self.alpha*self.mu_p*self.mu_n)/(3.0*4*self.M_p*self.M_n)*fdelta) * self.hc**3

        pot_em_np = pot_em_np.at[0].set(VC1_np)
        pot_em_np = pot_em_np.at[2].set(VMM_np)

        # neutron-neutron terms
        VMM_nn = (-(2*self.alpha*self.mu_n**2)/(3.0*4*self.M_n**2)*fdelta) * self.hc**3

        pot_em_nn = pot_em_nn.at[2].set(VMM_nn)

        return pot_em_pp, pot_em_np, pot_em_nn

    @partial(jit, static_argnums=(0,))
    def pionless_2b_contessi(self, rr):
        pot_2b=jnp.zeros(8)
        Cs_r =self.Cs*jnp.exp(-0.25*(rr/self.Rs)**2)
        Ct_r =self.Ct*jnp.exp(-0.25*(rr/self.Rt)**2)
        pot_2b = pot_2b.at[0].set( 3. * ( Cs_r + Ct_r ) )
        pot_2b = pot_2b.at[1].set( Cs_r - 3. * Ct_r )
        pot_2b = pot_2b.at[2].set( Ct_r - 3. * Cs_r )
        pot_2b = pot_2b.at[3].set( -1.*(Cs_r + Ct_r ))
        pot_2b = pot_2b/16.
        return pot_2b

    @partial(jit, static_argnums=(0,))
    def pionless_3b_contessi(self,rr):
        pot_3b=jnp.sqrt(self.D0)*jnp.exp(-0.25*(rr/self.R3)**2)
        return pot_3b

    @partial(jit, static_argnums=(0,))
    def argonne(self, rr):
        pot_2b = jnp.zeros(8)
        for i in range(6):
            pot_2b = pot_2b.at[i].set(jnp.interp(rr, self.r_tab, self.v_tab[i,:]))
        return pot_2b

    @partial(jit, static_argnums=(0,))
    def uix(self, rr):
        pot_3b = 0
        return pot_3b
