import numpy as np
import sys
from collections import defaultdict


data = defaultdict(list)
metric = None
for line in sys.stdin:
    if line:
        parts = line.split(' ')
        for part in parts:
            k, v = part.split('=')
            if k == 'metric':
                metric = v
            elif k == 'value':
                data[metric].append(float(v))

for metric, values in data.items():
    print('metric %s' % metric)
    print('samples %d' % len(values))
    print('mean %f' % np.mean(values))
    print('std %f' % np.std(values))
    print('min %f' % np.min(values))
    print('max %f' % np.max(values))
    print('')
