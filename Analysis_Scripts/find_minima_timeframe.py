import pandas as pd
import os

# Load the COLVAR file
fi = 'COLVAR'
data = pd.read_csv(fi, sep='\s+', comment="#", header=None)
data.columns = ['time', 'ref_coord_xy.x', 'ref_coord_xy.y', 'ref_coord_xy.z', 'Lig_coord.x', 'Lig_coord.y', 'Lig_coord.z', 'rel_x', 'rel_y', 'z', 'r', 'theta', 'up_restr_z.bias', 'up_restr_z.force2', 'low_restr_z.bias', 'low_restr_z.force2', 'restr_r.bias', 'restr_r.force2', 'metad.bias', 'metad.rbias', 'metad.rct']
print (data.head())

# Ask for user input for the target range of the CVs
cv1_min_max = input("Enter the target range for CV1 (z) in the format min,max: ")
target_cv1_min_default, target_cv1_max_default = cv1_min_max.split(',')
target_cv1_min_default = float(target_cv1_min_default)
target_cv1_max_default = float(target_cv1_max_default)

target_cv1_min = min(target_cv1_min_default, target_cv1_max_default)
target_cv1_max = max(target_cv1_min_default, target_cv1_max_default)

cv2_min_max = input("Enter the target range for CV2 (theta) in the format min,max: ")
target_cv2_min_default, target_cv2_max_default = cv2_min_max.split(',')
target_cv2_min_default = float(target_cv2_min_default)
target_cv2_max_default = float(target_cv2_max_default)

target_cv2_min = min(target_cv2_min_default, target_cv2_max_default)
target_cv2_max = max(target_cv2_min_default, target_cv2_max_default)

print ("Given target range for CV1 (z):", target_cv1_min, target_cv1_max)
print ("Given target range for CV2 (theta):", target_cv2_min, target_cv2_max)

# Find the time window where the CVs are within the target range
timepoints = []
for i in range(len(data)):
    if target_cv1_min <= data['z'][i] <= target_cv1_max and target_cv2_min <= data['theta'][i] <= target_cv2_max:
        timepoints.append(data['time'][i])

# Write the timepoints to a file with the z, theta, and bias potential values
outfile = f'timepoints_cv1_{target_cv1_min}_{target_cv1_max}_cv2_{target_cv2_min}_{target_cv2_max}.dat'
with open(outfile, 'w') as f:
    f.write(f"# Timepoints where CVs are within the target range cv1: {target_cv1_min} to {target_cv1_max} and cv2: {target_cv2_min} to {target_cv2_max}\n")
    f.write("# Time (ps) z theta bias_potential\n")
    for i in timepoints:
        f.write(f"{data['time'][i]:.3f} {data['z'][i]:.3f} {data['theta'][i]:.3f} {data['metad.bias'][i]:.3f}\n")

# Calculate the mid target range for CV1 and CV2
mid_target_cv1 = (target_cv1_min + target_cv1_max) / 2
mid_target_cv2 = (target_cv2_min + target_cv2_max) / 2

# Search for the nearest timepoint to the mid target range in data['z'] and data['theta']
target_timepoint = 0
min_diff1= 0.5 # Tolerance for the difference between the mid target range and the CV1
min_diff2 = 0.25 # Tolerance for the difference between the mid target range and the CV2
for i in range(len(data)):
    diff1 = abs(data['z'][i] - mid_target_cv1) 
    diff2 = abs(data['theta'][i] - mid_target_cv2)
    if diff1 < min_diff1 and diff2 < min_diff2:
        min_diff1 = diff1
        min_diff2 = diff2
        target_timepoint = data['time'][i]

target_timepoint = int(target_timepoint)
print(f"Nearest timepoint to the mid target range: {target_timepoint} ps")

# Remove the old index file if exists
if os.path.exists('new_index.ndx'):
    os.remove('new_index.ndx')
    
# Make a new index for Protein and Ligand
os.system("echo '1 | 13 | 14\nq' | gmx_mpi_d make_ndx -f production.tpr -o new_index.ndx")

# Run gmx to get the pdb at the target timepoint
os.system(f"gmx_mpi_d trjconv -f production.xtc -s production.tpr -o t_{target_timepoint}.pdb -n new_index.ndx -b {target_timepoint} -e {target_timepoint}")

