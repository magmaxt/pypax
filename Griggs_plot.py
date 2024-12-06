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
folder_name='241201-T1073-V2e-6_corrected/'

genpath2mesh = genpath + folder_name+ 'solution/solution-%s.pvtu'   
genpath2part = genpath + folder_name+'particles/particles-%s.pvtu'

exp_name = folder_name #'241103-T1473-V2e-16'

# Constants
sec_in_yr = 365*24*3600 # seconds in a year
sec_in_day = 24*3600 # seconds in a day


# Time steps
timesteps = np.array([67])

# theoretical strain rates
sampleH = 0.0006 #meter (half thickness of the sample)
V_shear = 2e-9 #2.0e-9 # shear velocity m/s
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
    # Print available data arrays
    print("mesh Point Data Arrays:", mesh.point_data.keys())
    print("mesh Cell Data Arrays:", mesh.cell_data.keys())
    print("mesh Field Data Arrays:", mesh.field_data.keys())

    path2part = genpath2part%timestep
    particles = pv.read(path2part)
    print("particles Point Data Arrays:", particles.point_data.keys())
    print("particles Cell Data Arrays:", particles.cell_data.keys())
    print("particles Field Data Arrays:", particles.field_data.keys())


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
    print(f'for sample mask: total {sum(sample_mask)} of filtered points ')

    #sample_mask = sample_mask * np.logical_and(sample.x > 0, sample.x < 0.0015)
    #sample_mask = sample_mask * np.logical_and(sample.x > 0, sample.x < 0.0015)
    
    #loc_threshold_r = sampleH/4 #5e-4 #5e-5
    # for i in range(5):
    #     sample_mask +=  ((sample.x - particles.points[i,0])**2+
    #                 (sample.y - particles.points[i,1])**2+
    #                 (sample.z - particles.points[i,2])**2 <= loc_threshold_r**2)            
    #     # sample_mask =  np.logical_or(sample_mask , 
    #     #         # np.logical_and(
    #     #         #     sample.data > 0.9,
    #     #             (sample.x - particles.points[i,0])**2+
    #     #             (sample.y - particles.points[i,1])**2+
    #     #             (sample.z - particles.points[i,2])**2 <= loc_threshold_r**2
    #     #         #)
    #     #     )
    #     print(f'for sample mask: total {sum(sample_mask)} of filtered points ')

    sample_purity = sample.copy()
    sample_purity.set_mask(sample_mask)
    
    stressII_masked = stressII.copy()
    strainrateII_masked = strainrateII.copy()

    stressII_masked.set_mask(sample_mask)
    strainrateII_masked.set_mask(sample_mask)

    
    #plotting particle locations
    figure = plt.figure(constrained_layout=True)
    figure.set_size_inches(3,3)
    ax = figure.add_gridspec(1,1)
    f1 = figure.add_subplot(ax[:,:])
    f1.yaxis.label.set_size(22)
    f1.xaxis.label.set_size(22)
    f1.set_xlabel('X',fontsize=22)
    f1.set_ylabel('Z',fontsize=22, color='k')
    for i in range(5):
        f1.plot(particles.points[i,0],particles.points[i,2],'.',label=f'{i}')
        # this show that a specific particle are not always listed at the same sequence
    #find middle particle
    ind_mid_particle = np.where(particles.points[:,0]==np.median(particles.points[:,0]))[0][0]
    f1.plot(particles.points[ind_mid_particle,0],particles.points[ind_mid_particle,2],'ko',label='mid',zorder=0)
    f1.legend()
    f1.grid()
    f1.set_xlim([-0.003,0.003])
    f1.set_ylim([-0.0045,0.0065])
    
    # get mid particle location data
    # particle_mask = sample.data > 0.9
    # loc_threshold = 5e-4 #5e-5
    # particle_mask = particle_mask * np.logical_and(
    #     abs(sample.x - particles.points[ind_mid_particle,0])<loc_threshold,
    #     abs(sample.y - particles.points[ind_mid_particle,1])<loc_threshold,
    #     abs(sample.z - particles.points[ind_mid_particle,2])<loc_threshold,
    #     )
 
    loc_threshold_r = sampleH/2 #5e-4 #5e-5
    particle_mask =  np.logical_and(sample.data > 0.9,
        (sample.x - particles.points[ind_mid_particle,0])**2+
        (sample.y - particles.points[ind_mid_particle,1])**2+
        (sample.z - particles.points[ind_mid_particle,2])**2 <= loc_threshold_r**2
        )
    print(f'for particle mask: total {sum(particle_mask)} of filtered points ')


    stressII_masked_particle = stressII.copy()
    strainrateII_masked_particle = strainrateII.copy()

    stressII_masked_particle.set_mask(particle_mask)
    strainrateII_masked_particle.set_mask(particle_mask)


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

T = 1073 # K
strainrate_theory_loc = A * stress_theory**n * np.exp(-Q/(R*T)) 
strainrate_theory_newtonian_loc = A * stress_theory**1 * np.exp(-Q/(R*T))  * 5e20

#eta = A**(-1/n) * np.exp(Q/(n*R*T)) * epsilon**((1-n)/n)    /2 # * 1e6

#mean for all points
# mean_stress=np.mean(stressII_masked.data)
# mean_strainrate=np.mean(strainrateII_masked.data)

#median for all points
mean_stress=np.mean(stressII_masked.data)
mean_strainrate=np.mean(strainrateII_masked.data)

# plotting particle locations
figure = plt.figure(constrained_layout=True)
figure.set_size_inches(9, 9)
ax = figure.add_gridspec(1, 1)
f2 = figure.add_subplot(ax[:, :])

# Axis labels and title
# Adjust tick number font size
f2.tick_params(axis='x', labelsize=22)  # X-axis tick font size
f2.tick_params(axis='y', labelsize=22)  # Y-axis tick font size

# Theory plot dislocation n=3.5
f2.loglog(stress_theory, strainrate_theory, 
          'r--', label='dislocation creep \n Zhang et al., 2006; T=1273K')
# Theory plot newtonian n=1
f2.loglog(stress_theory, strainrate_theory_newtonian, 
          'b--', label='diffusion creep; T=1273K')

# Theory plot dislocation n=3.5
f2.loglog(stress_theory, strainrate_theory_loc, 
          'r-', label='dislocation creep \n Zhang et al., 2006; T=1073K')
# Theory plot newtonian n=1
# f2.loglog(stress_theory, strainrate_theory_newtonian_loc, 
#           'b-', label='diffusion creep; T=1073K')

# Scatter plot with sample purity as the color
scatter = f2.scatter(
    stressII_masked.data, 
    strainrateII_masked.data, 
    c=sample_purity.data, 
    s=33, 
    cmap=plt.cm.magma, 
    zorder=2
)

# Add a colorbar
cbar = figure.colorbar(scatter, ax=f2, orientation='horizontal', shrink=0.3)
cbar.set_label('Sample Purity', fontsize=16)

# Additional scatter plots
#mean plot for all points
f2.scatter(mean_stress, mean_strainrate, s=2000, color='black', alpha=1, label='mean all', marker='*')
# plot for neighbors of the mid-particle
f2.scatter(
    stressII_masked_particle.data, 
    strainrateII_masked_particle.data,
    s=900, 
    color='red', 
    alpha=.9, 
    label='mid-particle neighbors', 
    marker='+', 
    zorder=3
)
# plot for mean of neighbors of the mid-particle
#mean for all points
mean_stress_mid_particle=np.mean(stressII_masked_particle.data)
mean_strainrate_mid_particle=np.mean(strainrateII_masked_particle.data)
f2.scatter(mean_stress_mid_particle, mean_strainrate_mid_particle,
            s=2222, color='red', alpha=.3, label='mean mid-parti', marker='o',zorder=2)


# Log scale and labels
f2.set_xscale('log')
f2.set_yscale('log')
f2.set_xlabel(r'$\sigma_{II}^s$ (Pa)', fontsize=22)
f2.set_ylabel(r'$\epsilon_{II}^s (s^{-1})$', fontsize=22)
f2.set_title(
    f'model: {exp_name} \n @ {(model_time[0] / sec_in_day):.1f} days and strain of {strain_theory[0]:.1f}\n' +
    f'mean strain rate: {mean_strainrate_mid_particle:.2e} ' +
    r'$\mathrm{s^{-1}}$' + f', mean stress: {mean_stress_mid_particle:.2e} Pa \n' +
    f'effective viscosity near mid-parti: {mean_stress_mid_particle/mean_strainrate_mid_particle:.1e} Pa*s'
    ,fontsize=16
)
# Legends, limits, and grid
f2.legend(fontsize=12,loc='lower right')
#f2.legend(fontsize=12,loc='upper left')

f2.set_xlim([1e8, 1e10])
f2.set_ylim([1e-7, 1e-5])
# f2.set_ylim([1e-7/10, 1e-5/10]) #V2e-5
# f2.set_ylim([1e-7*10, 1e-5*10]) #V2e-5

f2.grid()

# f2.set_xlim([1e8/10, 1e10*10])
# f2.set_ylim([1e-7/10, 1e-5*10])

# # Show the plot
# plt.show()
