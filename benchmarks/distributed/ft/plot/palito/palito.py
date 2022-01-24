from operator import ge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.interpolate import BSpline, make_interp_spline

plt.rcParams.update({'font.size': 12, 'font.family': 'Myriad Pro'})

# vit_threshold = [1.4e12, 1.2e12, 9e11, 6e11, 3e11, 1e11]
# vit_recovery = [126.012, 136.179, 147.53, 156.01, 171.71, 198.560]

vit_threshold = [1.4e12, 9e11, 6e11, 3e11, 1e11]
vit_recovery = [126.012, 147.53, 156.01, 171.71, 198.560]

bert_threshold = [5e11, 4e11, 1.5e11, 8e10, 5e10]
bert_recovery = [76.539, 76.547, 83.035, 160.836, 184.363]

plt.xlabel('recovery time (s)')
plt.ylabel('threshold (MB)')
# plt.xlim(7800, 8900)
# plt.ylim(0, 1)

linewidth=3

plt.plot(vit_recovery, vit_threshold, linewidth=linewidth, label="ViT Optimization", linestyle=':', 
            marker='o',markersize='5', markerfacecolor="black", markeredgewidth=0)
plt.plot(bert_recovery, bert_threshold, linewidth=linewidth, label="Bert Optimization", linestyle='--')

for i in range(len(vit_threshold)):
    plt.text(vit_recovery[i], vit_threshold[i], f"({vit_recovery[i]}, {vit_threshold[i]})",fontdict={'size':'6','color':'black'})
plt.legend()

plt.savefig("palito.svg", dpi=400)
plt.savefig("palito.png", dpi=400)