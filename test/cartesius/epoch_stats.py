import sys
import numpy as np
from collections import defaultdict

data = defaultdict(list)

epoch = 0
for i, line in enumerate(sys.stdin, 1):
    if i % 2 == 0:
        _, value = line.split(' ')
        data[epoch].append(float(value))
    else:
        _, value = line.split(' ')
        epoch = int(value)

for epoch, values in sorted(data.items(), key=lambda pair: pair[0]):
    print(epoch, 'min=%f max=%f mean=%f std=%f' % (np.min(values), np.max(values), np.mean(values), np.std(values)))
