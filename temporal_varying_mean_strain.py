import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from mesh import Volume

# Constants and paths
genpath = '/Users/tian_bc/Documents/2024-LC_delamination/Model_results/GriggsExp/Andromeda/241103/'
folder_name = '241103-T1273-V2e-16/'
genpath2mesh = genpath + folder_name + 'solution/solution-%s.pvtu'
genpath2part = genpath + folder_name + 'particles/particles-%s.pvtu'
exp_name = folder_name
sec_in_yr = 365 * 24 * 3600  # Seconds in a year

# Time steps to process
timesteps = np.arange(0, 100, 1)  # Example: time steps from 0 to 145 with step 10

# Initialize arrays to store results
time_steps = []
time_yrs = []  # Time in years
mean_strain_rates = []
mean_viscosities = []
strain_theory = []
strain_real_mean_epsilon = []
time = []
dt = []

# Iterate over time steps
for timestep in timesteps:
    print(f'Processing timestep: {timestep}')
    timestep_str = '{0:05d}'.format(timestep)
    path2mesh = genpath2mesh % timestep_str
    mesh = pv.read(path2mesh)

    # Extract model time in seconds
    model_time = mesh.field_data['TIME'][0]  # Time in seconds
    time_yrs.append(model_time / sec_in_yr)  # Convert to years
    
    sampleH = 0.0006
    V_shear = 2e-16
    epsilon_dot_theory = V_shear / (2 * sampleH)
    strain_theory.append(model_time * epsilon_dot_theory)

    # Extract relevant data
    points = np.array(mesh.points)
    stressII_array = np.array(mesh.point_data['stress_second_invariant'])
    strainrateII_array = np.array(mesh.point_data['strain_rate'])
    sample_array = np.array(mesh.point_data['sample'])

    # Filter points based on sample purity
    sample_mask = sample_array > 0.9
    filtered_stressII = stressII_array[sample_mask]
    filtered_strainrateII = strainrateII_array[sample_mask]

    # Compute mean strain rate and viscosity
    mean_stress = np.mean(filtered_stressII)
    mean_strainrate = np.mean(filtered_strainrateII)
    mean_viscosity = mean_stress / mean_strainrate

    # Store results
    time_steps.append(timestep)
    mean_strain_rates.append(mean_strainrate)
    mean_viscosities.append(mean_viscosity)


# Convert lists to numpy arrays for easier plotting
time_steps = np.array(time_steps)
time_yrs = np.array(time_yrs)
mean_strain_rates = np.array(mean_strain_rates)
mean_viscosities = np.array(mean_viscosities)
strain_theory=np.array(strain_theory)

# Plot Mean Strain Rate vs. Time Step and Time in Years
fig, ax1 = plt.subplots(figsize=(12, 6))

# Primary X-axis (Time Steps)
ax1.plot(time_steps, mean_strain_rates, 'bo-', label='Mean Strain Rate')
ax1.set_xlabel('Time Step', fontsize=14)
ax1.set_ylabel('Mean Strain Rate (s$^{-1}$)', fontsize=14)
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

# Secondary X-axis (Time in Years)
ax2 = ax1.twiny()  # Create a second X-axis on top
ax2.plot(time_yrs, mean_strain_rates, 'bo-', alpha=0)  # Invisible plot for alignment
ax2.set_xlabel('Time (Years)', fontsize=14)
ax2.tick_params(axis='x', labelcolor='r')

# Title and legend
plt.title('Mean Strain Rate vs. Time Step and Time in Years', fontsize=16)
plt.legend(loc='upper left')
plt.show()

# Plot Mean Viscosity vs. Time Step and Time in Years
fig, ax1 = plt.subplots(figsize=(12, 6))

# Primary X-axis (Time Steps)
ax1.plot(time_steps, mean_viscosities, 'ro-', label='Mean Viscosity')
ax1.set_xlabel('Time Step', fontsize=14)
ax1.set_ylabel('Mean Viscosity (Pa·s)', fontsize=14)
ax1.tick_params(axis='y', labelcolor='r')
ax1.grid(True)

# Secondary X-axis (Time in Years)
ax2 = ax1.twiny()  # Create a second X-axis on top
ax2.plot(time_yrs, mean_viscosities, 'ro-', alpha=0)  # Invisible plot for alignment
ax2.set_xlabel('Time (Years)', fontsize=14)
ax2.tick_params(axis='x', labelcolor='r')

# Title and legend
plt.title('Mean Viscosity vs. Time Step and Time in Years', fontsize=16)
plt.legend(loc='upper left')
plt.show()


# Plot Mean Strain Rate vs. Time Step and strain 
fig, ax1 = plt.subplots(figsize=(12, 6))

# Primary X-axis (Time Steps)
ax1.plot(time_steps, mean_strain_rates, 'bo-', label='Mean Strain Rate')
ax1.set_xlabel('Time Step', fontsize=14)
ax1.set_ylabel('Mean Strain Rate (s$^{-1}$)', fontsize=14)
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

# Secondary X-axis (Time in Years)
ax2 = ax1.twiny()  # Create a second X-axis on top
ax2.plot(strain_theory, mean_strain_rates, 'bo-', alpha=0)  # Invisible plot for alignment
ax2.set_xlabel('strain', fontsize=14)
ax2.tick_params(axis='x', labelcolor='r')

# Title and legend
plt.title('Mean Strain Rate vs. Time Step and Strain in Years', fontsize=16)
plt.legend(loc='upper left')
plt.show()