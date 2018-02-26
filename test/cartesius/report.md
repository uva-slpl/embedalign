Here you find a couple of scripts to extract short summaries from multiple runs of a model.


# Best validation AER

To get a summary of best validation AER you can run

```bash
srun tail -n1 ~/experiments/nibm1/dx128-dh128-adam/*/*/*/log | egrep -o 'aer=([0-9.]+)' | python aer_stats.py > ~/experiments/nibm1/dx128-dh128-adam/report.txt
```


# Average validation AER at an epoch


To get a summary of validation AER per epoch you can run, for example, 

```bash
egrep 'aer' ~/experiments/nibm1/handsards.en-fr/dx128-dh128-adam/1*/*/*/log | egrep -o 'Epoch [0-9]+|aer [0-9.]+' | python3 epochstats.py > ~/experiments/nibm1/handsards.en-fr/dx128-dh128-adam/epochs.txt
```

