import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# ##########
# SHIFT PLOT
# ##########

shifts = {}
rots = {}

for method in ['bayes', 'map']:
    with open(f"imgs/results/{method}/patch/params_used.json", 'w') as f:
        info = json.load(f)
        shifts[method] = info['shifts']
        rots[method] = info['rots']

with open("imgs/results/data/true_values.json", 'w') as f:
    info = json.load(f)
    shifts['true'] = info['shifts']
    rots['true'] = info['rots']





