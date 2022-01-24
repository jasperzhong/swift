import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.interpolate import BSpline, make_interp_spline

def get_x_cdfvalue(throughput, numbins=25):
    res_freq = stats.relfreq(throughput, numbins=numbins)
    x = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)
    cdf_value = np.cumsum(res_freq.frequency)
    # print(res_freq)
    # print(cdf_value)
    # print(x)
    #smooth
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, cdf_value, k=3)
    cdf_value_smooth = spl(x_smooth)
    return x_smooth, cdf_value_smooth

# load data
gc_throughput = []
with open("gc_throughput.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        gc_throughput.append(float(line))

gc_throughput = gc_throughput[1:]
gc_throughput = np.array(gc_throughput)

log_throughput = []
with open("log_throughput.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        log_throughput.append(float(line))

log_throughput = log_throughput[1:]
log_throughput = np.array(log_throughput)

log_x, log_cdf_value = get_x_cdfvalue(log_throughput)
gc_x, gc_cdf_value = get_x_cdfvalue(gc_throughput)
log_cdf_value.sort()
gc_cdf_value.sort()

plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.xlim(1000, 1200)
plt.ylim(0, 1)
plt.title('CDF')
  
plt.plot(log_x, log_cdf_value, linewidth=3)
plt.plot(gc_x, gc_cdf_value, linewidth=3)
plt.savefig("cdf.png", dpi=400)