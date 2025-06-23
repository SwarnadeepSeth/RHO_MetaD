import matplotlib.pyplot as plt
import numpy as np
import math
import sys
plt.style.use('custom_style')

def p_discrete(phi, rbias, intv):
    bins = list(intv)
    hist = [0 for i in range(len(bins)-1)]
    rbias_sums = [0 for i in range(len(bins)-1)]
    for i in range(len(phi)):
        if phi[i] < bins[0] or phi[i] > bins[-1]:
            continue
        bin = math.floor((phi[i]-bins[0])/phi_interval)
        hist[bin] += 1
        rbias_sums[bin] += rbias[i]
    hist = np.array(hist)
    prob = hist/sum(hist)
    rbias_sums = np.array(rbias_sums)
    rbias_avg = np.divide(rbias_sums, hist, where=hist!=0)
    new_p = np.multiply(prob, np.exp(beta*rbias_avg))
    new_p = new_p/sum(new_p)
    return new_p

def FES(p):
    FES = (-1/beta)*np.log(p, where=p!=0)
    FES = FES - min(FES)
    return FES

def PlotdelFofTime(z, theta, rbias, z1, theta1, z2, theta2, n_points=None, points=None, label=None, z_size=0.5, theta_size=0.5):
    if points:
        timepoints = points
    else:
        timepoints = np.linspace(0, len(z), n_points+1)[1:]
    dF = []
    for i in timepoints:
        sample_z = z[:int(i)]
        sample_theta = theta[:int(i)]
        sample_rbias = rbias[:int(i)]
        f = delF_two_vars(sample_z, sample_theta, sample_rbias, z1, theta1, z2, theta2, z_size, theta_size)
        dF.append(f)
    timepoints += starting_point
    dF = np.nan_to_num(dF, nan=np.nanmax(dF)+1)
    plt.plot(np.array(timepoints) / 1000, dF, color='xkcd:green', label='ΔF(StateB - StateA)')
    plt.xlabel('time (ns)')
    plt.ylabel('ΔF (kJ/mol)')
    fe_range = abs(max(dF)-min(dF))
    if label:
        plt.legend(loc='upper right')
    plt.axhline(dF[-1], color='xkcd:red', linestyle='--')
    xtext = ((timepoints[-1] - starting_point) / 1000) * 0.75 + (starting_point / 1000)
    plt.text(xtext, dF[-1] + 0.025*fe_range, str(round(dF[-1], 2)) + ' kJ/mol', fontsize=24)
    #plt.title(f'ΔF from (z={z1}, θ={theta1}) to (z={z2}, θ={theta2})')

def delF_two_vars(z, theta, rbias, z1, theta1, z2, theta2, z_size, theta_size):
    '''StateB - StateA'''
    # Define regions around minima
    frames_A = [(z[i], theta[i], idx) for idx, i in enumerate(range(len(z))) 
                if z[i] >= z1-z_size/2 and z[i] <= z1+z_size/2 
                and theta[i] >= theta1-theta_size/2 and theta[i] <= theta1+theta_size/2]
    avg_rbias_A = 0 if len(frames_A) == 0 else np.mean([rbias[i[2]] for i in frames_A])
    w_A = np.exp(beta*avg_rbias_A)
    
    frames_B = [(z[i], theta[i], idx) for idx, i in enumerate(range(len(z))) 
                if z[i] >= z2-z_size/2 and z[i] <= z2+z_size/2 
                and theta[i] >= theta2-theta_size/2 and theta[i] <= theta2+theta_size/2]
    avg_rbias_B = 0 if len(frames_B) == 0 else np.mean([rbias[i[2]] for i in frames_B])
    w_B = np.exp(beta*avg_rbias_B)
    
    weighted_A = len(frames_A)*w_A if len(frames_A) > 0 else 1
    weighted_B = len(frames_B)*w_B if len(frames_B) > 0 else 1
    f = (-1/beta)*np.log(weighted_B/weighted_A) + unbound_corr
    return f

# Parameters
fi = 'COLVAR'
T = 300
kb = 0.0083144621  # kJ/mol/K
beta = 1/(kb*T)
phi_interval = 0.1
pace = 500  # 1ps
z_size = 0.5
theta_size = 0.5
unbound_corr = 0  # kJ/mol
files = sys.argv[1].split(',')  # COLVAR files
starting_point = 0
starting_point *= 1000

# Read data
zs = []
thetas = []
rbiass = []
for f in files:
    s = open(f, "r").readlines()[:-1]
    datlines = [x for x in s if not x.startswith("#")][:-2]
    vars = s[0].split()[2:]  # Skip '#! FIELDS time'
    z_idx = vars.index('z')
    theta_idx = vars.index('theta')
    rbias_idx = vars.index('metad.rbias')
    datlines = [i for i in s if not i.startswith('#')]
    z = [float(i.split()[z_idx]) for i in datlines[starting_point:]]
    theta = [float(i.split()[theta_idx]) for i in datlines[starting_point:]]
    rbias = [float(i.split()[rbias_idx]) for i in datlines[starting_point:]]
    zs.append(z)
    thetas.append(theta)
    rbiass.append(rbias)

print ("Enter the point locations (z, theta) separated by commas:")
print ("Your input: ", end="")
print(sys.argv)  # ['convergence_FES.py', 'COLVAR', 'z', '2,-3,4,-3']

# Get the point locations from sys.argv[2]
z1 = float(sys.argv[2].split(',')[0])
theta1 = float(sys.argv[2].split(',')[1])
z2 = float(sys.argv[2].split(',')[2])
theta2 = float(sys.argv[2].split(',')[3])

# Plot for each file
for i, f in enumerate(files):
    lname = f.replace('/', '_').split('.')[0]
    PlotdelFofTime(zs[i], thetas[i], rbiass[i], 
                   z1, theta1, z2, theta2,
                   n_points=100, z_size=z_size, theta_size=theta_size, label=lname)
    plt.savefig(f'FE_time_{lname}_z_theta.png', dpi=500)
    plt.savefig(f'FE_time_{lname}_z_theta.svg')
    plt.clf() 

# Plot Z and Theta in Time
for i, f in enumerate(files):
    lname = f.replace('/', '_').split('.')[0]
    plt.plot(np.array(range(len(zs[i]))) / 1000, zs[i], label=lname, color='xkcd:royal blue')
    plt.xlabel('time (ns)')
    plt.ylabel('Z (nm)')
    plt.savefig(f'z_time_{lname}.png', dpi=500)
    plt.savefig(f'z_time_{lname}.svg')
    plt.clf()

    plt.plot(np.array(range(len(thetas[i]))) / 1000, thetas[i], label=lname, color='xkcd:maroon')
    plt.xlabel('time (ns)')
    plt.ylabel(r'$\theta$ (radians)')
    plt.savefig(f'theta_time_{lname}.png', dpi=500)
    plt.savefig(f'theta_time_{lname}.svg')
    plt.clf()