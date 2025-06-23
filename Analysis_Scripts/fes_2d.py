import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import sys
from PIL import Image
import glob
import os

def p_discrete_2d(phi1, phi2, rbias, intv1, intv2):
    bins1 = list(intv1)
    bins2 = list(intv2)
    phi_interval1 = bins1[1] - bins1[0]
    phi_interval2 = bins2[1] - bins2[0]

    hist = np.zeros((len(bins1)-1, len(bins2)-1))
    weight_sum = np.zeros((len(bins1)-1, len(bins2)-1))

    for i in range(len(phi1)):
        if phi1[i] < bins1[0] or phi1[i] > bins1[-1] or phi2[i] < bins2[0] or phi2[i] > bins2[-1]:
            continue
        bin1 = min(math.floor((phi1[i] - bins1[0]) / phi_interval1), len(bins1)-2)
        bin2 = min(math.floor((phi2[i] - bins2[0]) / phi_interval2), len(bins2)-2)
        hist[bin1, bin2] += 1
        weight_sum[bin1, bin2] += np.exp(beta * rbias[i])  # per-sample reweighting

    prob_reweighted = weight_sum / np.sum(weight_sum)
    return prob_reweighted

def FES_2d(prob_reweighted):
    FES = np.full_like(prob_reweighted, np.nan)
    with np.errstate(divide='ignore'):
        FES = (-1 / beta) * np.log(prob_reweighted, where=prob_reweighted > 0)
    FES = np.nan_to_num(FES, nan=np.nanmax(FES))
    FES -= np.min(FES)
    return FES

def PlotFES_2d(phi1, phi2, rbias, intv1, intv2, label=None, plot_range1=None, plot_range2=None, cbar=True):
    p = p_discrete_2d(phi1, phi2, rbias, intv1, intv2)
    #replace zero counts with the minimum value of the rest
    FES_phi = FES_2d(p)
    # Write as a csv file
    df = pd.DataFrame(FES_phi, columns=intv2[:-1], index=intv1[:-1])
    df.to_csv('FES.csv')

    if plot_range1:
        idx1_min = np.searchsorted(intv1, plot_range1[0])
        idx1_max = np.searchsorted(intv1, plot_range1[1])
        intv1 = intv1[idx1_min:idx1_max+1]
        FES_phi = FES_phi[idx1_min:idx1_max, :]
    
    if plot_range2:
        idx2_min = np.searchsorted(intv2, plot_range2[0])
        idx2_max = np.searchsorted(intv2, plot_range2[1])
        intv2 = intv2[idx2_min:idx2_max+1]
        FES_phi = FES_phi[:, idx2_min:idx2_max]

    X, Y = np.meshgrid(intv1[:-1] + phi_interval1/2, intv2[:-1] + phi_interval2/2)
    plt.contourf(X, Y, FES_phi.T, levels=50, cmap='viridis')
    if cbar:
        plt.colorbar(label='FES (kJ/mol)')
    if label:
        plt.title(label)
    if flip:
        plt.gca().invert_xaxis()
    plt.xlabel(var1)
    plt.ylabel(var2)

def PlotFESofTime_2d(phi1, phi2, rbias, intv1, intv2, label=None, n_points=None, points=None, plot_range1=None, plot_range2=None, output_file='FES_time.gif'):
    if points:
        timepoints = points
    else:
        timepoints = np.linspace(0, len(phi1), n_points*2)[-n_points:]
        timepoints = [int(i) for i in timepoints]
    for i in range(len(timepoints)):
        plt.clf()
        sample_phi1 = phi1[:int(timepoints[i])]
        sample_phi2 = phi2[:int(timepoints[i])]
        sample_rbias = rbias[:int(timepoints[i])]
        timelab = 'Time: ' + str(int(timepoints[i]/pace/2)) + 'ns'
        if label:
            timelab = label + '               ' + timelab
        PlotFES_2d(sample_phi1, sample_phi2, sample_rbias, intv1,\
                    intv2, label=timelab, plot_range1=plot_range1,\
                      plot_range2=plot_range2)
        plt.savefig(f'{out_frame_dir}/{i}.png')
    make_gif(out_frame_dir, output_file)
        

def make_gif(frame_folder,out):
    def sort_int(f):
        return int(f.split('.')[0].split('/')[-1])
    imgs = glob.glob(f"{frame_folder}/*.png")
    imgs.sort(key=sort_int)
    frames = [Image.open(image) for image in imgs]
    frame_one = frames[0]
    frame_one.save(out, format="GIF", append_images=frames,
               save_all=True, duration=1000/FPS, loop=0)

# Load data from files
fi = 'COLVAR'
T = 300
kb = 0.0083144621  # kJ/mol/K
beta = 1/kb/T
phi_interval1 = 0.25 # 1 # Interval for first collective variable
phi_interval2 = 1.0 # 0.2  # Interval for second collective variable
pace = 500
FPS = 3
out_frame_dir = 'Video'
if not os.path.exists(out_frame_dir):
    os.makedirs(out_frame_dir)
f = sys.argv[1]

var1 = sys.argv[2]
var2 = sys.argv[3]
flip = var1[0] == '-'
var1 = var1[1:] if flip else var1

s = open(f, "r").readlines()[:-1]
plot_range1 = (float(sys.argv[4].split(',')[0]), float(sys.argv[4].split(',')[1])) if len(sys.argv) > 4 else None
plot_range2 = (float(sys.argv[5].split(',')[0]), float(sys.argv[5].split(',')[1])) if len(sys.argv) > 5 else None
datlines = [x for x in s if not x.startswith("#")][:-2]
vars = s[0].split()[2:]
var1_index = vars.index(var1)
var2_index = vars.index(var2)
rbias_idx = vars.index('metad.rbias')
datlines = [i for i in s if not i.startswith('#')]
phi1 = [float(i.split()[var1_index]) for i in datlines]
phi2 = [float(i.split()[var2_index]) for i in datlines]
rbias = [float(i.split()[rbias_idx]) for i in datlines]
low1 = min(phi1) if not plot_range1 else plot_range1[0]
high1 = max(phi1) if not plot_range1 else plot_range1[1]
low2 = min(phi2) if not plot_range2 else plot_range2[0]
high2 = max(phi2) if not plot_range2 else plot_range2[1]
intv1 = np.arange(low1, high1, phi_interval1)
intv2 = np.arange(low2, high2, phi_interval2)
PlotFES_2d(phi1, phi2, rbias, intv1, intv2, label=f, plot_range1=plot_range1, plot_range2=plot_range2)
outname = f.replace("/", "_") + '_FES.png'
plt.savefig(outname, dpi=500)
plt.clf()

low1 = min(phi1) if not plot_range1 else plot_range1[0]
high1 = max(phi1) if not plot_range1 else plot_range1[1]
low2 = min(phi2) if not plot_range2 else plot_range2[0]
high2 = max(phi2) if not plot_range2 else plot_range2[1]
intv1 = np.arange(low1, high1, phi_interval1)
intv2 = np.arange(low2, high2, phi_interval2)
PlotFESofTime_2d(phi1, phi2, rbias, intv1, intv2, n_points=50,\
                  plot_range1=plot_range1, label = f,
                      plot_range2=plot_range2, output_file=outname.replace('.png', '.gif'))
