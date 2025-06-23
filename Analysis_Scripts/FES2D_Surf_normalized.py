import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import filters, generate_binary_structure, binary_erosion, minimum_filter
from scipy.interpolate import griddata
from heapq import heappop, heappush
from matplotlib.colors import BoundaryNorm, ListedColormap

# Arial
plt.rcParams['font.family'] = 'Arial'

def read_data(filename):
    df = pd.read_csv(filename, index_col=0)
    z = df.columns.astype(float)
    theta = df.index.astype(float)
    free_energy = df.values
    return theta, z, free_energy

# # Detect local minima in 2D array
# def detect_local_minima(arr):
#     # Define neighborhood structure (2D)
#     neighborhood = generate_binary_structure(2, 2)

#     # Apply minimum filter to find local minima
#     local_min = (minimum_filter(arr, footprint=neighborhood) == arr)

#     # Identify the background (non-local minima)
#     background = (arr == np.max(arr))

#     # Remove background to only keep local minima
#     eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
#     detected_minima = local_min ^ eroded_background

#     # Return the indices of local minima
#     return np.where(detected_minima) 


def detect_local_minima(arr, min_separation=1.9):
    """
    Detect local minima in a 2D array, ensuring minima are approximately min_separation pixels apart.
    
    Parameters:
    arr : ndarray
        2D input array (e.g., image or grid).
    min_separation : float
        Approximate minimum distance (in pixels) between detected minima (default: 10).
    
    Returns:
    tuple
        (row_indices, col_indices) of detected local minima.
    """
    # Create a circular footprint for minimum filter (diameter ~ 2 * min_separation + 1)
    radius = int(min_separation)
    size = 2 * radius + 1  # Diameter
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    footprint = x*x + y*y <= radius*radius  # Circular mask

    # Apply minimum filter with large circular footprint
    local_min = (minimum_filter(arr, footprint=footprint, mode='nearest') == arr)

    # Identify the background (non-local minima, max value)
    neighborhood = generate_binary_structure(2, 2)  # 8-connected for background
    background = (arr == np.max(arr))

    # Remove background to only keep local minima
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_minima = local_min & ~eroded_background

    # Return the indices of local minima
    return np.where(detected_minima)

# Implement Dijkstra's algorithm to find the shortest path between two minima
def dijkstra(free_energy, start, goal):
    rows, cols = free_energy.shape
    visited = set()
    pq = [(0, start)]  # Priority queue with (cost, position)
    dist = {start: 0}
    prev = {start: None}

    while pq:
        current_cost, current = heappop(pq)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            break  # Stop once we reach the goal

        # Check neighbors (up, down, left, right)
        neighbors = [(current[0] + i, current[1] + j) 
                     for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                     if 0 <= current[0] + i < rows and 0 <= current[1] + j < cols]

        for neighbor in neighbors:
            if neighbor in visited:
                continue
            cost = current_cost + abs(free_energy[neighbor] - free_energy[current])
            if neighbor not in dist or cost < dist[neighbor]:
                dist[neighbor] = cost
                prev[neighbor] = current
                heappush(pq, (cost, neighbor))

    # Reconstruct the path
    path = []
    step = goal
    while step is not None:
        path.append(step)
        step = prev[step]
    path.reverse()
    return path


# ==============================================================================
# Read the data
theta, z, free_energy = read_data('FES.csv')
free_energy = free_energy.T

# Create a grid for interpolation
theta_grid, z_grid = np.meshgrid(theta, z)

# Interpolate the data =========================================================
def interpolate_data(theta, z, free_energy):
    # Flatten the grids for interpolation
    points = np.array([(th, zz) for th in theta for zz in z])
    values = free_energy.flatten()

    # Create a finer grid for smoother interpolation
    theta = np.linspace(min(theta), max(theta), 100)
    z = np.linspace(min(z), max(z), 100)
    theta_grid, z_grid = np.meshgrid(theta, z)

    # Interpolate the data
    free_energy = griddata(points, values, (theta_grid, z_grid), method='linear')

    # Smooth the interpolated data
    #free_energy = filters.gaussian_filter(free_energy, sigma=1)

    return theta, z, theta_grid, z_grid, free_energy

# ==============================================================================
# Find local minima in the 2D free energy surface by sorting FES values
minimas = detect_local_minima(free_energy)

# Create a list to store minima values (using theta_grid and z_grid, not theta and z)
minima_values = []
for i in range(len(minimas[0])):
    minima_values.append([theta_grid[minimas[0][i], minimas[1][i]],  # Use theta_grid
                          z_grid[minimas[0][i], minimas[1][i]],     # Use z_grid
                          free_energy[minimas[0][i], minimas[1][i]]])  # Free energy value

# Create a DataFrame for sorting minima
minima_df = pd.DataFrame(minima_values, columns=['Theta', 'Z', 'Free Energy']).sort_values('Free Energy')

# Print the local minima
print('Local Minimum in Free Energy ==============================')
print(minima_df)

# ==============================================================================
# Select two local minima for Dijkstra's algorithm from minima_df
start_min = tuple(minima_df.iloc[0][['Theta', 'Z']].values)
goal_min = tuple(minima_df.iloc[1][['Theta', 'Z']].values)

print("="*80)
print('Start Minima:', start_min)
print('Goal Minima:', goal_min)

# Ensure the coordinates are represented as indices in the free energy array
start_min_idx = np.unravel_index(np.argmin(np.abs(theta_grid - start_min[0]) + np.abs(z_grid - start_min[1])), free_energy.shape)
goal_min_idx = np.unravel_index(np.argmin(np.abs(theta_grid - goal_min[0]) + np.abs(z_grid - goal_min[1])), free_energy.shape)

print('Start Minima Index:', start_min_idx)
print('Goal Minima Index:', goal_min_idx)

# Find the shortest energy path between the two minima
shortest_path = dijkstra(free_energy, start_min_idx, goal_min_idx)
print ('Shortest Path:', shortest_path)

# Find the free energy barrier along the shortest path
free_energy_barrier = [free_energy[pt[0], pt[1]] for pt in shortest_path]
print('Free Energy Barrier along the Shortest Path:', free_energy_barrier)
highest_barier = round(max(free_energy_barrier),2)
print('highest Barrier:', highest_barier, 'kJ/mol')

# ==============================================================================
# Plotting the smoothed surface (heatmap)
plt.figure(figsize=(12, 8))
cp = plt.contourf(theta_grid, z_grid, free_energy, cmap='viridis')
cbar = plt.colorbar(cp, label='Free Energy (kJ/mol)')
cbar.ax.tick_params(labelsize=20)
plt.xlabel(r'CV2 ($\theta$)', fontsize=24)
plt.ylabel('CV1 (Z)', fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=20)

# Plot the local minima
plt.scatter(minima_df['Theta'], minima_df['Z'], color='red', s=100, label='Local Minima')

# Plot the shortest path
path_theta = [theta_grid[pt[0], pt[1]] for pt in shortest_path]
path_z = [z_grid[pt[0], pt[1]] for pt in shortest_path]
plt.plot(path_theta, path_z, color='magenta', lw=2, label='Shortest Path')

# Write the highest barrier on the plot
plt.text(0.5, 0.9, f'Barrier: {highest_barier} kJ/mol', fontsize=20, transform=plt.gca().transAxes)

plt.tight_layout()
plt.savefig('FES_2D.png')
plt.savefig('FES_2D.svg')
plt.show()

# Plotting the smoothed surface (3D)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta_grid, z_grid, free_energy, cmap='viridis')
# theta_fine, z_fine, z_grid_fine, theta_grid_fine, free_energy_fine = interpolate_data(theta, z, free_energy)
# ax.plot_surface(theta_grid_fine, z_grid_fine, free_energy_fine, cmap='viridis') 

ax.view_init(elev=30, azim=45)
ax.set_xlabel(r'CV2 ($\theta$)', fontsize=16)
ax.set_ylabel('CV1 (Z)', fontsize=16)
ax.set_zlabel('Free Energy (kJ/mol)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)

# Plot the local minima
ax.scatter(minima_df['Theta'], minima_df['Z'], minima_df['Free Energy'], color='red', s=100, label='Local Minima')

# Plot the shortest path in 3D
ax.plot(path_theta, path_z, [free_energy[pt[0], pt[1]]-1 for pt in shortest_path], color='magenta', lw=5, label='Shortest Path')

# Write the highest barrier on the plot
ax.text2D(0.5, 0.7, f'Barrier: {highest_barier} kJ/mol', fontsize=20, transform=ax.transAxes)

ax.legend(fontsize=20, frameon=False)   
plt.tight_layout()
plt.savefig('FES_2D_3D.png', bbox_inches='tight', dpi=300)

# Save two projections of the 3D plot
ax.view_init(elev=0, azim=0)
plt.savefig('FES_2D_3D_1.png', bbox_inches='tight', dpi=300)
ax.view_init(elev=0, azim=90)
plt.savefig('FES_2D_3D_2.png', bbox_inches='tight', dpi=300)
plt.close()

# =================================================================================================================
vmax = 150

# Limit minima by vmax
minima_df = minima_df[minima_df['Free Energy'] <= vmax]
print ("Minima that are below vmax ==============================")
print (minima_df)

# ==============================================================================
# Normalize FES: Subtract M2:M10 average
# if len(minima_df) >= 10:
#     m2_to_m10_avg = minima_df['Free Energy'].iloc[2:13].mean()
# else:
#     print(f"Warning: Only {len(minima_df)} minima found.")
#     m2_to_m10_avg = minima_df['Free Energy'].iloc[1:].mean() if len(minima_df) > 1 else 0.0
# print(f"M2:M10 average: {m2_to_m10_avg:.2f} kJ/mol")
# normalized_free_energy = free_energy - m2_to_m10_avg

# Get minima which are below 10 kJ/mol and calculate the average 
minima_below_10 = minima_df[minima_df['Free Energy'] < 10]
average_minima_below_10 = minima_below_10['Free Energy'].mean()

minima_above_10 = minima_df[minima_df['Free Energy'] >= 10]
average_minima_above_10 = minima_above_10['Free Energy'].mean()

print ("Average of Minima below 10 kJ/mol ==============================")
print (average_minima_below_10)

print ("Average of Minima above 10 kJ/mol ==============================")
print (average_minima_above_10)

# Normalize FES: Subtract average of minima above 10 kJ/mol
normalized_free_energy = free_energy - average_minima_above_10

print ("Normalized Free Energy ==============================")
# 2D Plot with metadyn_normalized.py style
plt.figure(figsize=(10, 10))

vmin = np.min(normalized_free_energy)

# Mask z outside [2, 8]
z_mask = (z_grid < 2) | (z_grid > 8)
normalized_free_energy[z_mask] = np.nan

# Clip values > vmax
normalized_free_energy[normalized_free_energy > vmax] = vmax

# Custom colormap: Yellowish jet to hot pink
jet = plt.colormaps.get_cmap('jet')
yellowish_colors = jet(np.linspace(0, 0.65, 256))  # Up to yellow
hot_pink = np.array([[1.0, 0.713, 0.756, 1.0]])  # RGBA for hot pink
combined_colors = np.vstack([yellowish_colors, hot_pink])
custom_cmap = ListedColormap(combined_colors)
boundaries = np.linspace(vmin, 101, 258)  # 0–100 yellowish, 100+ hot pink
norm = BoundaryNorm(boundaries, custom_cmap.N)

smoothing = True
if smoothing:
    # Sophisticated Gaussian smoothing with increased grid density
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import griddata
    from scipy.interpolate import RBFInterpolator
    from scipy.interpolate import CloughTocher2DInterpolator
    from scipy.spatial.distance import pdist

    # Increase grid density
    theta_fine = np.linspace(theta_grid.min(), theta_grid.max(), 512)
    z_fine = np.linspace(z_grid.min(), z_grid.max(), 512)
    theta_grid_fine, z_grid_fine = np.meshgrid(theta_fine, z_fine)

    # Prepare data for RBF
    points = np.vstack((theta_grid.ravel(), z_grid.ravel())).T
    values = normalized_free_energy.ravel()
    valid_mask = ~np.isnan(values)  # Handle NaN from prior masking
    points = points[valid_mask]
    values = values[valid_mask]

    # Normalize theta for periodicity (-pi to pi)
    theta_norm = ((points[:, 0] + np.pi) % (2 * np.pi)) - np.pi
    points[:, 0] = theta_norm

    # Compute dynamic epsilon as median distance between points
    distances = pdist(points, metric='euclidean')
    epsilon = np.median(distances)  # Median distance for balanced smoothing

    # Fit RBF interpolator
    rbf = RBFInterpolator(points, values, kernel='multiquadric', smoothing=1.0, epsilon=10*epsilon)
    #rbf = RBFInterpolator(points, values, kernel='cubic', smoothing=0.1)

    # Evaluate RBF on fine grid
    fine_points = np.vstack((theta_grid_fine.ravel(), z_grid_fine.ravel())).T
    fine_points[:, 0] = ((fine_points[:, 0] + np.pi) % (2 * np.pi)) - np.pi
    smoothed_free_energy = rbf(fine_points).reshape(theta_grid_fine.shape)

    # Apply Gaussian smoothing to polish
    smoothed_free_energy = gaussian_filter(smoothed_free_energy, sigma=1.0, mode='wrap')  # Strong smoothing
    smoothed_free_energy = gaussian_filter(smoothed_free_energy, sigma=0.5, mode='wrap')  # Polishing pass

    # Clip values > vmax
    smoothed_free_energy[smoothed_free_energy > vmax] = vmax

    # Plot contours
    sc = plt.contourf(theta_grid_fine, z_grid_fine, smoothed_free_energy, 30, cmap=custom_cmap, norm=norm)
    plt.contour(theta_grid_fine, z_grid_fine, smoothed_free_energy, 2, colors='black', linewidths=1)

else:
    # Plot contours
    sc = plt.contourf(theta_grid, z_grid, normalized_free_energy, 30, cmap=custom_cmap, norm=norm)
    plt.contour(theta_grid, z_grid, normalized_free_energy, 2, colors='black', linewidths=1)

# Colorbar
cbar = plt.colorbar(sc)
cbar.ax.set_ylabel('Free Energy (kJ/mol)', fontsize=30)
cbar.ax.tick_params(labelsize=24)

# Set ticks to include minima, maxima, and key intermediates
min_val = np.nanmin(smoothed_free_energy)
max_val = np.nanmax(smoothed_free_energy)
ticks = np.round(np.linspace(min_val, max_val, 10), 1)  # 6 ticks, rounded for clarity
ticks = np.unique(np.append(ticks, [0, 100]))  # Ensure 0 and 100 are included
cbar.set_ticks(ticks)

# Plot minima as stars
#plt.scatter(minima_df['Theta'], minima_df['Z'], c='white', s=150, marker='*',
#            edgecolors='black', label='Local Minima', zorder=10)


plt.scatter(minima_below_10['Theta'], minima_below_10['Z'], c='white', s=500, marker='*',
            edgecolors='black', label='Local Minima', zorder=10)

# Plot shortest path
path_theta = [theta_grid[pt[0], pt[1]] for pt in shortest_path]
path_z = [z_grid[pt[0], pt[1]] for pt in shortest_path]
plt.plot(path_theta, path_z, color='magenta', lw=2, label='Shortest Path')

# Write the highest barrier on the plot
#plt.text(0.3, 0.83, f'Barrier (A-B): {highest_barier} kJ/mol', fontsize=24, transform=plt.gca().transAxes)

# Annotate A and B to the start_min and the goal minima
plt.annotate('I', (path_theta[0], path_z[0]), xytext=(path_theta[0] + 0.4, path_z[0] + 0.4),
             arrowprops=dict(facecolor='white', shrink=0.05), fontsize=60, color='white')

plt.annotate('II', (path_theta[-1], path_z[-1]), xytext=(path_theta[-1] + 0.4, path_z[-1] + 0.4),
             arrowprops=dict(facecolor='white', shrink=0.05), fontsize=60, color='white')

plt.xlabel(r'CV1 ($\theta$)', fontsize=32)
plt.ylabel('CV2 (Z)', fontsize=32)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tick_params(axis='both', which='minor', labelsize=24)
plt.xlim(theta.min(), theta.max())
plt.ylim(1.98, 8.02)

# Increase the tick length
plt.tick_params(axis='both', which='major', length=10, width=2)
# Increase the tick width
plt.tick_params(axis='both', which='minor', length=5, width=1)

# Darken the plot borders (spines)
for spine in ax.spines.values():
    spine.set_edgecolor('black')   # Border color
    spine.set_linewidth(3.5)       # Border thickness

plt.legend(fontsize=24, frameon=False)
plt.tight_layout()
plt.savefig('FES_2D_normalized.png', dpi=300)
plt.savefig('FES_2D_normalized.svg', dpi=600)
plt.show()
#plt.close()

# ==============================================================================
# Plot the KDE distribution of minima
plt.figure(figsize=(10, 6))
#plt.hist(minima_df['Free Energy'], bins=30, density=True, alpha=0.9, color='xkcd:blue', label='Minima Distribution')

# Red bar plot for minima below 10 kJ/mol
plt.bar(minima_below_10['Free Energy'],
         height=1/len(minima_below_10), width=0.5, color='red', alpha=0.7, label='Minima < 10 kJ/mol', align='center')

# Green bar plot for minima above 10 kJ/mol
plt.bar(minima_above_10['Free Energy'],
         height=1/len(minima_above_10), width=0.5, color='green', alpha=0.7, label='Minima >= 10 kJ/mol', align='center')

#plt.axvline(average_minima_below_10, color='red', linestyle='--', label='Avg Minima < 10 kJ/mol')
#plt.axvline(average_minima_above_10, color='green', linestyle='--', label='Avg Minima >= 10 kJ/mol')

# Write the difference between the two averages
difference = -(average_minima_above_10 - average_minima_below_10)
plt.text(0.5, 0.7, f'Difference: {difference:.2f} kJ/mol', fontsize=20, transform=plt.gca().transAxes)
plt.xlabel('Minima Free Energy (kJ/mol)', fontsize=24)
plt.ylabel('Normalized Count', fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend(fontsize=18, frameon=False)
plt.tight_layout()
plt.savefig('minima_distribution.svg')
plt.show()