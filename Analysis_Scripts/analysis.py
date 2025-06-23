import matplotlib.pyplot as plt
import numpy as np
import sys

s = open(sys.argv[1], "r").readlines()[:-1]
datlines = [x for x in s if not x.startswith("#")][:-2]
fields = ['time'] + sys.argv[2:-1]
blocksize = int(sys.argv[-1])
prefix = sys.argv[1] + '_'
vars = s[0].split()
field_index = [vars.index(f)-2 for f in fields]
dat = {}
for i,f in enumerate(fields):
      dat[f] = [float(x.split()[field_index[i]]) for x in datlines]
blockdat = {}
for f in fields:
      blockdat[f] = []
for i in range(1, len(dat[fields[0]]), blocksize):
      for f in fields:
            blockdat[f].append(sum(dat[f][i:i+blocksize])/len(dat[f][i:i+blocksize]))
blockdat[fields[0]] = [x for x in range(0, len(dat[fields[0]])-1, blocksize)]
# rest = [blockdat[fields[1]][i] < 5.5 and blockdat[fields[2]][i] < 0.6 for i in range(len(blockdat[fields[0]]))]
for f in fields[1:]:
      plt.plot(blockdat[fields[0]], blockdat[f], label=f)
      # plt.plot(blockdat[fields[0]], rest, label='within walls?', color='red')
      plt.xlabel("time (ps)")
      plt.ylabel(f)
      plt.legend()
      plt.savefig(prefix + f + ".png")
      plt.clf()

#print standard deviation of phi and psi

for f in fields[1:]:
      print(f + " std dev: ", np.std(dat[f]))
