import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from mesh import Volume

# Constants and paths (unchanged)
genpath = '/Users/tian_bc/Documents/2024-LC_delamination/Model_results/GriggsExp/Andromeda/241103/'
folder_name = '241103-T1273-V2e-16/'
genpath2mesh = genpath + folder_name + 'solution/solution-%s.pvtu'
genpath2part = genpath + folder_name + 'particles/particles-%s.pvtu'
exp_name = folder_name
sec_in_yr = 365 * 24 * 3600
sec_in_day = 24 * 3600
timesteps = np.array([150])
sampleH = 0.0006
V_shear = 2e-16
epsilon_dot_theory = V_shear / (2 * sampleH)

# Iterative reading of the timesteps
nod = len(timesteps)

for i in range(nod):
    print('i is:', i)
    timestep = '{0:05d}'.format(timesteps[i])
    path2mesh = genpath2mesh % timestep
    mesh = pv.read(path2mesh)
    path2part = genpath2part % timestep
    particles = pv.read(path2part)

    # Process the mesh
    array_names = mesh.array_names
    model_time = mesh.field_data['TIME']
    strain_theory = epsilon_dot_theory * model_time
    print(f"Model time is: {int(model_time[0])} s and {int(model_time[0] / sec_in_yr / 1e3)} kyrs")
    bounds = mesh.bounds
    points = np.array(mesh.points)
    stressII_array = np.array(mesh.point_data['stress_second_invariant'])
    strainrateII_array = np.array(mesh.point_data['strain_rate'])
    sample_array = np.array(mesh.point_data['sample'])

    stressII = Volume(points, stressII_array)
    strainrateII = Volume(points, strainrateII_array)
    sample = Volume(points, sample_array)

    # Compute and process the mesh data
    sample_mask = sample.data > 0.850
    sample_purity = sample.copy()
    sample_purity.set_mask(sample_mask)
    stressII_masked = stressII.copy()
    strainrateII_masked = strainrateII.copy()
    stressII_masked.set_mask(sample_mask)
    strainrateII_masked.set_mask(sample_mask)

    # Plotting particle locations
    figure = plt.figure(constrained_layout=True)
    figure.set_size_inches(3, 3)
    ax = figure.add_gridspec(1, 1)
    f1 = figure.add_subplot(ax[:, :])
    f1.yaxis.label.set_size(22)
    f1.xaxis.label.set_size(22)
    f1.set_xlabel('X', fontsize=22)
    f1.set_ylabel('Z', fontsize=22, color='k')
    # List of colors
    colors = ['red', 'orange', 'green', 'blue',  'purple']
    for i in range(5):
        f1.plot(particles.points[i, 0], particles.points[i, 2], 'o', label=f'{i}', color=colors[i % len(colors)],markersize=12)
    ind_mid_particle = np.where(particles.points[:, 0] == np.median(particles.points[:, 0]))[0][0]
    f1.plot(particles.points[ind_mid_particle, 0], particles.points[ind_mid_particle, 2], 'kv', label='mid', zorder=5)
    f1.legend()
    f1.grid()
    f1.set_xlim([-0.003, 0.003])
    f1.set_ylim([-0.0045, 0.0065])

    stressII_masked_particle = stressII.copy()
    strainrateII_masked_particle = strainrateII.copy()
    # stressII_masked_particle0 = stressII.copy()
    # strainrateII_masked_particle0 = strainrateII.copy()
    # stressII_masked_particle1 = stressII.copy()
    # strainrateII_masked_particle1 = strainrateII.copy()
    # stressII_masked_particle2 = stressII.copy()
    # strainrateII_masked_particle2 = strainrateII.copy()
    # stressII_masked_particle3 = stressII.copy()
    # strainrateII_masked_particle3 = strainrateII.copy()
    # stressII_masked_particle4 = stressII.copy()
    # strainrateII_masked_particle4 = strainrateII.copy()
    stressII_masked_particle_list = []
    strainrateII_masked_particle_list =[]
    for i in range(5):
        stressII_masked_particle_list.append(stressII.copy())
        strainrateII_masked_particle_list.append(strainrateII.copy())

    loc_threshold_r = sampleH / 5
    particle_mask = np.logical_and(sample.data > 0.85,
                                   (sample.x - particles.points[ind_mid_particle, 0]) ** 2 +
                                   (sample.y - particles.points[ind_mid_particle, 1]) ** 2 +
                                   (sample.z - particles.points[ind_mid_particle, 2]) ** 2 <= loc_threshold_r ** 2)
    print(f'for particle mask: total {sum(particle_mask)} of filtered points ')
    stressII_masked_particle.set_mask(particle_mask)
    strainrateII_masked_particle.set_mask(particle_mask)

    particle_mask_list = []
    for i in range(5):
        particle_mask_list.append(
            np.logical_and(sample.data > 0.85,
                                   (sample.x - particles.points[i, 0]) ** 2 +
                                   (sample.y - particles.points[i, 1]) ** 2 +
                                   (sample.z - particles.points[i, 2]) ** 2 <= loc_threshold_r ** 2)
        )
        print(f'for particle mask {i}: total {sum(particle_mask_list[i])} of filtered points ')
        stressII_masked_particle_list[i].set_mask(particle_mask_list[i])
        strainrateII_masked_particle_list[i].set_mask(particle_mask_list[i])

    # particle_mask0 = np.logical_and(sample.data > 0.85,
    #                                (sample.x - particles.points[0, 0]) ** 2 +
    #                                (sample.y - particles.points[0, 1]) ** 2 +
    #                                (sample.z - particles.points[0, 2]) ** 2 <= loc_threshold_r ** 2)
    # print(f'for particle mask 0: total {sum(particle_mask0)} of filtered points ')
    # stressII_masked_particle0.set_mask(particle_mask0)
    # strainrateII_masked_particle0.set_mask(particle_mask0)

    # particle_mask1 = np.logical_and(sample.data > 0.85,
    #                                (sample.x - particles.points[1, 0]) ** 2 +
    #                                (sample.y - particles.points[1, 1]) ** 2 +
    #                                (sample.z - particles.points[1, 2]) ** 2 <= loc_threshold_r ** 2)
    # print(f'for particle mask 1: total {sum(particle_mask1)} of filtered points ')
    # stressII_masked_particle1.set_mask(particle_mask1)
    # strainrateII_masked_particle1.set_mask(particle_mask1)

    # particle_mask2 = np.logical_and(sample.data > 0.85,
    #                                (sample.x - particles.points[2, 0]) ** 2 +
    #                                (sample.y - particles.points[2, 1]) ** 2 +
    #                                (sample.z - particles.points[2, 2]) ** 2 <= loc_threshold_r ** 2)
    # print(f'for particle mask 2: total {sum(particle_mask2)} of filtered points ')
    # stressII_masked_particle2.set_mask(particle_mask2)
    # strainrateII_masked_particle2.set_mask(particle_mask2)

    # particle_mask3 = np.logical_and(sample.data > 0.85,
    #                                (sample.x - particles.points[3, 0]) ** 2 +
    #                                (sample.y - particles.points[3, 1]) ** 2 +
    #                                (sample.z - particles.points[3, 2]) ** 2 <= loc_threshold_r ** 2)
    # print(f'for particle mask 3: total {sum(particle_mask3)} of filtered points ')
    # stressII_masked_particle3.set_mask(particle_mask3)
    # strainrateII_masked_particle3.set_mask(particle_mask3)

    # particle_mask4 = np.logical_and(sample.data > 0.85,
    #                                (sample.x - particles.points[4, 0]) ** 2 +
    #                                (sample.y - particles.points[4, 1]) ** 2 +
    #                                (sample.z - particles.points[4, 2]) ** 2 <= loc_threshold_r ** 2)
    # print(f'for particle mask 4: total {sum(particle_mask4)} of filtered points ')
    # stressII_masked_particle4.set_mask(particle_mask4)
    # strainrateII_masked_particle4.set_mask(particle_mask4)



    # Flow rule theoretical parameters
    A = 1e-23
    n = 3.5
    Q = 310000  # J/mol
    R = 8.314  # J/(mol K)
    T = 1273  # K
    stress_theory = np.logspace(0, 20, num=21, base=10, endpoint=True)
    strainrate_theory = A * stress_theory ** n * np.exp(-Q / (R * T))
    strainrate_theory_newtonian = A * stress_theory ** 1 * np.exp(-Q / (R * T)) * 5e15
    T = 1073
    strainrate_theory_loc = A * stress_theory ** n * np.exp(-Q / (R * T))
    T = 1473
    strainrate_theory_loc_1473 = A * stress_theory ** n * np.exp(-Q / (R * T))

    mean_stress = np.mean(stressII_masked.data)
    mean_strainrate = np.mean(strainrateII_masked.data)

    # Plotting particle locations
    figure = plt.figure(constrained_layout=True)
    figure.set_size_inches(9, 9)
    ax = figure.add_gridspec(1, 1)
    f2 = figure.add_subplot(ax[:, :])
    f2.tick_params(axis='x', labelsize=22)
    f2.tick_params(axis='y', labelsize=22)

    # Theory plot dislocation n=3.5
    f2.loglog(stress_theory, strainrate_theory, 'r--', label='dislocation creep \n Zhang et al., 2006; T=1273K')
    f2.loglog(stress_theory, strainrate_theory_newtonian, 'b--', label='diffusion creep; T=1273K')
    f2.loglog(stress_theory, strainrate_theory_loc, 'm-', label='dislocation creep \n Zhang et al., 2006; T=1073K')
    f2.loglog(stress_theory, strainrate_theory_loc_1473, 'r-', label='dislocation creep \n Zhang et al., 2006; T=1473K')

    # Scatter plot with sample purity as the color
    scatter = f2.scatter(stressII_masked.data, strainrateII_masked.data, c=sample_purity.data, s=33, cmap=plt.cm.magma, zorder=2)
    cbar = figure.colorbar(scatter, ax=f2, orientation='horizontal', shrink=0.3)
    cbar.set_label('Sample Purity', fontsize=16)

    # Additional scatter plots
    f2.scatter(mean_stress, mean_strainrate, s=2000, color='black', alpha=1, label='mean all', marker='*')
    f2.scatter(stressII_masked_particle.data, strainrateII_masked_particle.data, s=900, color='red', alpha=.9, label='mid-particle neighbors', marker='+', zorder=3)
    mean_stress_mid_particle = np.mean(stressII_masked_particle.data)
    mean_strainrate_mid_particle = np.mean(strainrateII_masked_particle.data)
    f2.scatter(mean_stress_mid_particle, mean_strainrate_mid_particle, s=2222, color='red', alpha=.3, label='mean mid-parti', marker='o', zorder=2)

    for i in range(5):
        f2.scatter(stressII_masked_particle_list[i].data, strainrateII_masked_particle_list[i].data, s=200, color=colors[i], alpha=.9, label='particle'+str(i)+'neighbors', marker='v', zorder=3)
    


    # Log scale and labels
    f2.set_xscale('log')
    f2.set_yscale('log')
    f2.set_xlabel(r'$\sigma_{II}^s$ (Pa)', fontsize=22)
    f2.set_ylabel(r'$\epsilon_{II}^s (s^{-1})$', fontsize=22)
    f2.set_title(f'model: {exp_name} \n @ {(model_time[0] / sec_in_yr / 1e3):.0f} kyrs and strain of {strain_theory[0]:.1f}\n' +
                 f'mean strain rate: {mean_strainrate_mid_particle:.2e} ' +
                 r'$\mathrm{s^{-1}}$' + f', mean stress: {mean_stress_mid_particle:.2e} Pa \n' +
                 f'effective viscosity near mid-parti: {mean_stress_mid_particle / mean_strainrate_mid_particle:.1e} Pa*s', fontsize=16)
    f2.legend(fontsize=12, loc='lower right')
    f2.set_xlim([1e6, 1e8])
    f2.set_ylim([1e-14, 1e-12])
    f2.grid()

   # 3D Plot with distance to theoretical curve
log_stress = np.log10(stressII_masked.data)
log_strainrate = np.log10(strainrateII_masked.data)
log_stress_theory = np.log10(stress_theory)
log_strainrate_theory = np.log10(strainrate_theory)

# Calculate distances from the theoretical curve
# distances = np.abs(np.interp(log_stress, log_stress_theory, log_strainrate_theory) - log_strainrate)
# distance_norm = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
distance_norm = np.abs(np.interp(log_stress, log_stress_theory, log_strainrate_theory) - log_strainrate)

# 3D Scatter Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# Use a high-contrast colormap like 'viridis' or 'plasma'
sc = ax.scatter(
    points[sample_mask, 0], 
    points[sample_mask, 1], 
    points[sample_mask, 2], 
    c=distance_norm, 
    cmap='viridis',  # Use 'viridis' or 'plasma' for better contrast
    marker='o',
    s=5
)
# Add colorbar
cbar = plt.colorbar(sc, ax=ax, label='Proximity to Theoretical Line')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Filtered 3D Sample Points (Purity > 0.85, Distance to Theory)')
plt.show()


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# Use a high-contrast colormap like 'viridis' or 'plasma'
sc = ax.scatter(
    points[sample_mask, 0], 
    points[sample_mask, 1], 
    points[sample_mask, 2], 
    c=log_strainrate, 
    cmap='viridis',  # Use 'viridis' or 'plasma' for better contrast
    marker='o',
    s=1
)
# Add colorbar
cbar = plt.colorbar(sc, ax=ax, label='log_strainrate')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Filtered 3D Sample Points (Purity > 0.85, strain rate)')
plt.show()

# Cross-section plot at Y=0
fig, ax = plt.subplots(figsize=(8, 6))
cross_section_mask = np.abs(points[sample_mask, 1]) < 1e-5  # Points close to Y=0
# Scatter plot for the cross-section
sc = ax.scatter(
    points[sample_mask][cross_section_mask, 0], 
    points[sample_mask][cross_section_mask, 2], 
    c=distance_norm[cross_section_mask], 
    cmap='viridis',  # Use 'viridis' or 'plasma' for better contrast
    marker='o',
    s=10
)
# Add colorbar
cbar = plt.colorbar(sc, ax=ax, label='Proximity to Theoretical Line')
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_title('Cross-Section at Y=0')
plt.show()

# Cross-section plot at Y=0
fig, ax = plt.subplots(figsize=(8, 6))
cross_section_mask = np.abs(points[sample_mask, 1]) < 1e-4  # Points close to Y=0
# Scatter plot for the cross-section
sc = ax.scatter(
    points[sample_mask][cross_section_mask, 0], 
    points[sample_mask][cross_section_mask, 2], 
    c=log_strainrate[cross_section_mask], 
    cmap='viridis',  # Use 'viridis' or 'plasma' for better contrast
    marker='o',
    s=10
)
# Add colorbar
cbar = plt.colorbar(sc, ax=ax, label='strain rate')
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_title('Cross-Section at Y=0')
plt.show()

# Cross-section plot at Y=0.0003
fig, ax = plt.subplots(figsize=(8, 6))
cross_section_mask = np.abs(points[sample_mask, 1]-0.0003) < 1e-4  # Points close to Y=0
# Scatter plot for the cross-section
sc = ax.scatter(
    points[sample_mask][cross_section_mask, 0], 
    points[sample_mask][cross_section_mask, 2], 
    c=distance_norm[cross_section_mask], 
    cmap='viridis',  # Use 'viridis' or 'plasma' for better contrast
    marker='o'
)
# Add colorbar
cbar = plt.colorbar(sc, ax=ax, label='Proximity to Theoretical Line')
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_title('Cross-Section at Y=0.0003')
plt.show()


# Cross-section plot X-Y with all projected
fig, ax = plt.subplots(figsize=(8, 6))
# Scatter plot for the cross-section
sc = ax.scatter(
    points[sample_mask, 0], 
    points[sample_mask, 1], 
    c=distance_norm, 
    cmap='viridis',  # Use 'viridis' or 'plasma' for better contrast
    marker='o'
)
# Add colorbar
cbar = plt.colorbar(sc, ax=ax, label='Proximity to Theoretical Line')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Cross-Section X-Y with all Z')
plt.show()