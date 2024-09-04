import numpy as np
import matplotlib.pyplot as plt

freq_30 = [ 31.25      ,   62.5       ,    93.75 ]
freq_amps_30 = {
                'occ_1': [5.70747391e-07, 2.10245525e-06, 2.76565329e-07],
                'occ_2': [8.66950963e-07, 3.53606241e-06, 5.05181865e-07]
                }

freq_20_8 = [20.83333333  ,  41.66666667   , 62.5       ,    83.33333333   , 104.16666667]
freq_amps_20_8 = {
                'occ_1': [2.83381712e-06, 1.36342562e-06, 2.05375939e-06, 8.32245125e-07, 8.98439163e-07],
                'occ_2': [3.14768543e-06, 1.58877468e-06, 5.46307993e-06, 1.55938988e-06, 1.00037834e-06]
                }

f1 = plt.figure()
f2 = plt.figure()

ax1 = f1.add_subplot(111)
ax1.set_title("Frequency plot (30Hz)")
ax1.set_xlabel("Frequency bins")
ax1.set_ylabel("Frequency amplitude")

ax2 = f2.add_subplot(111)
ax2.set_title("Frequency plot (20.8Hz)")
ax2.set_xlabel("Frequency bins")
ax2.set_ylabel("Frequency amplitude")

ax1.stem(freq_30, freq_amps_30['occ_1'], 'b', label='occ_1')
ax1.stem(freq_30, freq_amps_30['occ_2'], 'g', label='occ_2')

ax2.stem(freq_20_8, freq_amps_20_8['occ_1'], 'b', label='occ_1')
ax2.stem(freq_20_8, freq_amps_20_8['occ_2'], 'g', label='occ_2')

plt.legend
plt.show()