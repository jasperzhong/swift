from operator import ge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.interpolate import BSpline, make_interp_spline

plt.rcParams.update({'font.size': 12, 'font.family': 'Myriad Pro'})

def get_x_cdfvalue(throughput, numbins=15):
    res_freq = stats.relfreq(throughput, numbins=numbins)
    x = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)
    cdf_value = np.cumsum(res_freq.frequency)
    # print(res_freq)
    # print(cdf_value)
    # print(x)
    #smooth
    x_smooth = np.linspace(x.min(), x.max(), 200)
    spl = make_interp_spline(x, cdf_value, k=3)
    cdf_value_smooth = spl(x_smooth)

    # for tokens
    x_smooth = x_smooth
    return x_smooth, cdf_value_smooth

def load_data(file):
    array = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            array.append(float(line))
    
    array = array[1:-1]
    array = np.array(array)
    return array

# load data
gc_throughput = load_data("gc.txt")
replica = load_data("replica.txt")
print(replica)


replica_x, replica_cdf_value = get_x_cdfvalue(replica)
gc_x, gc_cdf_value = get_x_cdfvalue(gc_throughput)


replica_cdf_value.sort()
gc_cdf_value.sort()


plt.xlabel('throughput (imgs/s)')
plt.ylabel('CDF')
plt.xlim(7800, 8900)
plt.ylim(0, 1)
# plt.title('Bert CDF')

linewidth=3

plt.plot(gc_x, gc_cdf_value, linewidth=linewidth, label="global checkpoint")
plt.plot(replica_x, replica_cdf_value, linewidth=linewidth, label="Replication")


plt.legend()

plt.savefig("cdf.svg", dpi=400)
plt.savefig("cdf.png", dpi=400)