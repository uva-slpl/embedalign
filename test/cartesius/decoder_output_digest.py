import numpy as np
import sys

if len(sys.argv) < 2:
    config = 'default'
else:
    config = sys.argv[1]

L = []
Px = []
Py = []
X = []
Y = []
AER = []
dataset = None
# dataset        loss    perp_x    perp_y     acc_x     acc_y       AE
for i, line in enumerate(sys.stdin, 1):
    if i % 3 == 0:
        dataset, loss, perp_x, perp_y, acc_x, acc_y, aer = line.split()
        L.append(float(loss))
        Px.append(float(perp_x))
        Py.append(float(perp_y))
        X.append(float(acc_x))
        Y.append(float(acc_y))
        AER.append(float(aer))


print('dataset runs config loss-mean loss-std perp-x-mean perp-x-std perp-y-mean perp-y-std x-mean x-std y-mean y-std aer-mean aer-std')
print('%s %d %s %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f ' % (
    dataset, 
    len(X), 
    config,
    np.mean(L), np.std(L), 
    np.mean(Px), np.std(Px),
    np.mean(Py), np.std(Py),
    np.mean(X) * 100, np.std(X) * 100, 
    np.mean(Y) * 100, np.std(Y) * 100, 
    np.mean(AER) * 100, np.std(AER) * 100))
