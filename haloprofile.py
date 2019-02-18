import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt


# I generated HMF data from a package online, just for testing.
# The HMF used is Sheth-Tormen.  
hmf_data = np.loadtxt('mvec.txt')


rho0 = 2.78e11 # rho0 in Msun/h and Mpc/h units
omega_m = 0.3
del_crit = 1.686
del_halo = 200*del_crit # This is taken to be the halo density, since we usually define 
                        # r_vir as that enclosing 200x the critical density. 

rho_halo = del_halo*rho0*omega_m # and here the average halo density is defined. 

def r_vir(m):
    # The average halo density is used to convert from halo mass to r_vir, because the
    # integration will be till r_vir for power. 
    m = np.array(m)
    return np.power(3.0*m/(4*np.pi*rho_halo),1.0/3)

def conc(m):
    # Conc param taken at z = 0. 
    m = np.array(m)/(2e12)
    
    return 14.85*np.power(m,-0.084)

def ein(r,m):
    # Einasto profile. Takes in an array of radii for which to compute density. 
    r = np.array(r)
    
    alpe = 0.17
    alpha = 0.17
    rs = r_vir(m)/conc(m)
    
    density_prof = np.exp((-2.0/alpe)*(np.power(1.0*r/rs,alpha) - 1.0))

    unnorm_mass = np.trapz(density_prof,r)

    return m*density_prof/unnorm_mass

def dm_power(m):
    
    # Total power ignoring the 4pi factor and doing just the radial integral. 
    
    c = 3e8
    sigv = 3e-25
    mdm = 34e-4
    
    rads = np.linspace(0,r_vir(m),1000)
    
    
    pwr = c*c*sigv/mdm*np.power(ein(rads,m),2)
    
    total = np.trapz(pwr*rads*rads,rads)
    
    return total


masses = hmf_data[:,0]
logmasses = np.log10(masses)
dndlogm = hmf_data[:,7]

pwr = np.zeros(len(masses))
for i,m in enumerate(masses):
    pwr[i] = dm_power(masses[i])
    
pwr = np.array(pwr)
    
integrand = pwr*dndlogm # This is P(M)*dn(M)/dlogM
dlogm = np.diff(logmasses)[0]

total_power = np.zeros(len(masses))

for i,m in enumerate(masses):
    
    # This loop takes a slice of the integrand, starting from the minimum mass
    # we want to consider for the total power. 
    # It then integrates from M_min to M = 10^15 the integrand P(M)*dn/dlogM.
    arr = integrand[i:]
    total_power[i] = np.trapz(arr,dx=dlogm)
    
plt.loglog(masses,total_power)
plt.xlabel(r'$M_{min}$' + ' ' + r'(Mpc/h)')
plt.ylabel('Total Power')
plt.title('Halo Power Convergence')
