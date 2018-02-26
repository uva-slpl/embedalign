import numpy as np
import sys

data = []
for line in sys.stdin:
    if line:
        parts = line.split(' ')
        for part in parts:
            k, v = part.split('=')
            if k == 'val_aer':
                data.append(float(v))

print('samples %d' % len(data))
print('mean %f' % np.mean(data))
print('std %f' % np.std(data))
print('min %f' % np.min(data))
print('max %f' % np.max(data))
