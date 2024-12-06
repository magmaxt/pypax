"""
Created on Fri Oct 11 11:15:18 2024, Boston, MA, USA

@Processing ASPECT output with Python
@authors: Alexandre JANIN + Xiaochuan Tian
"""

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from mesh import Volume



#genpath = '/home/alexandre/Alexandre/BibliothekaAlexandrina/11-Griggs-X-A/ASPECT/RUNs/241103-T1473-V2e-16/'
genpath = '/Users/tian_bc/Documents/2024-LC_delamination/Model_results/GriggsExp/Peloton_lc/241201/'
folder_name='241201-T1273-V2e-6_corrected/'

genpath2mesh = genpath + folder_name+ 'solution/solution-%s.pvtu'   
genpath2part = genpath + folder_name+'particles/particles-%s.pvtu'

exp_name = folder_name #'241103-T1473-V2e-16'

# Constants
sec_in_yr = 365*24*3600 # seconds in a year
sec_in_day = 24*3600 # seconds in a day


# Time steps
timesteps = np.array([70])

# theoretical strain rates
sampleH = 0.0006 #meter (half thickness of the sample)
V_shear = 2.0e-9 # shear velocity m/s
epsilon_dot_theory = V_shear/(2*sampleH)
# theoretical strain





# --- Iterative reading of the timesteps

nod = len(timesteps)

for i in range(nod):

    # -------- READ DATA --------

    # Read data from the pvtu file into a variable named mesh, which contains
    # all information needed for the plot.

    timestep = '{0:05d}'.format(timesteps[i])

    path2mesh = genpath2mesh%timestep
    mesh = pv.read(path2mesh)

    path2part = genpath2part%timestep
    particles = pv.read(path2part)

    # -------- PROCESS THE MESH --------

    # To retrieve the keys (array names) of the pvtu file
    array_names = mesh.array_names

    # TO access model time
    model_time = mesh.field_data['TIME']
    type(model_time)
    # calculate theoretical strain
    strain_theory = epsilon_dot_theory * model_time
    #print("model time is:", model_time[0], 's','and',model_time[0]/sec_in_day,'days')
    print(f"Model time is: {int(model_time[0])} s and {int(model_time[0] /sec_in_day)} days")

    # To find the spatial bounds of the domain
    bounds = mesh.bounds

    # To get the coordinates of the points
    points = np.array(mesh.points)

    # To get the data at the points ( have to be in 'array_names')
    stressII_array     = np.array(mesh.point_data['stress_second_invariant'])
    strainrateII_array = np.array(mesh.point_data['strain_rate'])
    sample_array       = np.array(mesh.point_data['sample'])

    stressII     = Volume(points,stressII_array)
    strainrateII = Volume(points,strainrateII_array)
    sample       = Volume(points,sample_array)



    # -------- COMPUTE AND PROCESS THE MESH DATA --------

    sample_mask = sample.data > 0.90
    #sample_mask = sample_mask * np.logical_and(sample.x > 0, sample.x < 0.0015)
    sample_mask = sample_mask * np.logical_and(sample.x > 0, sample.x < 0.0015)

    sample_purity = sample.copy()
    sample_purity.set_mask(sample_mask)
    
    stressII_masked = stressII.copy()
    strainrateII_masked = strainrateII.copy()

    stressII_masked.set_mask(sample_mask)
    strainrateII_masked.set_mask(sample_mask)

# -------- FIGURE --------
# flow rule theoretical parameters:
A = 1e-23 ##1e-2 MPa^-n/s is equivalent to 1e-23 Pa^-n/s for n=3.5#
n = 3.5
Q = 310000 # J/mol
R = 8.314 # J/(mol K)

T = 1273 # K
stress_theory = np.logspace(0,20,num=21,base=10,endpoint=True) 
strainrate_theory = A * stress_theory**n * np.exp(-Q/(R*T)) 

strainrate_theory_newtonian = A * stress_theory**1 * np.exp(-Q/(R*T))  * 5e20
#eta = A**(-1/n) * np.exp(Q/(n*R*T)) * epsilon**((1-n)/n)    /2 # * 1e6

#mean
mean_stress=np.mean(stressII_masked.data)
mean_strainrate=np.mean(strainrateII_masked.data)

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.set_title(
    f'model: {exp_name} \n @ {int(model_time[0] / sec_in_day)} days and strain of {strain_theory[0]:.1f}\n'
    + f'mean strain rate: {mean_strainrate:.2e} ' + r'$\mathrm{s^{-1}}$' + f', mean stress: {mean_stress:.2e} Pa'
)

# theory plot dislocation n=3.5
ax.loglog(stress_theory,strainrate_theory,'r--',label='dislocation creep from Zhang et al., 2006')
# theory plot newtonian n=1
ax.loglog(stress_theory,strainrate_theory_newtonian,'b--',label='diffusion creep')

cmap = ax.scatter(stressII_masked.data,strainrateII_masked.data,c=sample_purity.data,s=33,cmap=plt.cm.magma,zorder=2)
ax.scatter(mean_stress,mean_strainrate,s=100,color='red',alpha=1,label='mean')






cbar = fig.colorbar(cmap,orientation='vertical',shrink=0.5,label='sample purity')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\sigma_{II}^s$ (Pa)')
ax.set_ylabel(r'$\epsilon_{II}^s (s^{-1})$')
ax.legend(fontsize=9)
fig.tight_layout()




# plt.xlim([np.mean(stressII_masked.data)*0.1,np.mean(stressII_masked.data)*10])
# plt.ylim([np.mean(strainrateII_masked.data)*0.1,np.mean(strainrateII_masked.data)*10])

plt.xlim([1e8,1e10])
plt.ylim([1e-7,1e-5])

# # more relaxed
# plt.xlim([1e7,1e11])
# plt.ylim([1e-8,1e-4])
plt.grid()

plt.show()





