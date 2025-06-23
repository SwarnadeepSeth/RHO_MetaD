import pandas as pd
import MDAnalysis as mda
from MDAnalysis.coordinates.core import writer

# List of timepoints from .dat file
timepoint_file = pd.read_csv("timepoints_cv1_5.9_6.1_cv2_-1.28_-0.9.dat", comment="#", sep='\s+')
timepoints = timepoint_file.iloc[:, 0].values
timepoints = [int(timepoint) for timepoint in timepoints]

# Take every 50th frame
timepoints = timepoints[::2]

# Load the trajectory
u = mda.Universe("production.tpr", "production.xtc")

# Convert timepoints to frame indices based on trajectory time step (dt)
dt = u.trajectory.dt  # Time between frames in ps
frame_indices = [int(t / dt) for t in timepoints]

# Select the group of interest (e.g., Protein + Ligand)
selection = u.select_atoms("protein or resname LIG")  # Adjust selection as needed

# Write all frames into a single PDB
with mda.Writer("combined_frames.pdb", selection.n_atoms) as pdb_writer:
    for frame_index in frame_indices:
        u.trajectory[frame_index]  # Move to the specific frame
        pdb_writer.write(selection)

print("Combined PDB file written to 'combined_frames.pdb'")
