import numpy as np
import sys
#TODO!!! for GAP!!!
if len(sys.argv) < 2:
    config = 'default'
else:
    config = sys.argv[1]

GAP = []
dataset = "test.lst"
# dataset        loss    perp_x    perp_y     acc_x     acc_y       AE
for i, line in enumerate(sys.stdin, 1):
    if 'MEAN_GAP' in line: 
        name, gap = line.split() # TODO long split maybe use regex
        GAP.append(float(gap))


print('dataset runs config gap-mean gap-std')
print('%s %d %s %.2f %.2f ' % (
    dataset,
    len(GAP),
    config,
    np.mean(GAP) * 100, np.std(GAP) * 100))
