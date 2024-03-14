import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

SSVEP_THRESHOLD = 10 ** -5

def linearly_regress_plot_data(amps):
    amps = list(map(float, amps))
    no_of_data_points = len(amps)
    xs_flicker = np.arange(0, no_of_data_points).tolist()
    
    reg = LinearRegression(fit_intercept=True)
    reg.fit(np.array(xs_flicker).reshape(-1,1), np.array(amps))

    for t in range(no_of_data_points):
        amps_drift = reg.coef_[0] * (no_of_data_points - t)
        # print(f"amps[{t}]: {amps[t]}")
        amps[t] = amps[t] + amps_drift - reg.intercept_
        # print(f"regressed amps[{t}]: {amps[t]}")

    return list(map(str, amps))

def proportions_for_each_element(l):
    s = set(l)
    list_len = len(l)
    for distinct_el in s:
        no_of_apparitions = len(list(filter(lambda el: el == distinct_el, l)))
        proportion = no_of_apparitions / list_len * 100
        print(f"{distinct_el}: {proportion:.2f}%")


fig, ax = plt.subplots()
ax.set_title("Hemisphere amplitudes graph")
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
t = None
y_min = -1
y_max = -1
abnormal_amps_indices = []

if sys.argv[1] == 'yes':
    lr = True
else:
    lr = False

for file_path in sys.argv[2:]:
    if '\\' in file_path:
        hemisphere = file_path.split('\\')[-1].split('.')[0]
    else: # '/' in file_path
        hemisphere = file_path.split('/')[-1].split('.')[0]

    with open(file_path) as f:
        lines = f.readlines()
        if t is None:
            t = [line.split(' ')[1] for line in lines]
            # t = list(map(int, t))
        amps = [line.split(' ')[0] for line in lines]
        # amps = list(map(float, amps))

    if y_min == -1:
        y_min = min(amps)
    else:
        y_min = min(y_min, min(amps))
    if y_max == -1:
        y_max = max(amps)
    else:
        y_max = max(y_min, max(amps))

    amps = list(map(float, amps))
    for i in range(len(amps) - 1):
        if amps[i] - amps[i+1] >= SSVEP_THRESHOLD:
           abnormal_amps_indices.append(i+1)
    for i in range(len(abnormal_amps_indices) - 1):
        abnormal_amps_indices[i] = abnormal_amps_indices[i+1] - abnormal_amps_indices[i]
    abnormal_amps_indices = abnormal_amps_indices[:-1]
    # print(f"abnormal_amps_indices_differences: {abnormal_amps_indices}")
    # proportions_for_each_element(abnormal_amps_indices)
    amps = list(map(str, amps))

    if lr: amps = linearly_regress_plot_data(amps)
    plt.plot(t, amps, label=hemisphere)

    # Debugging
    # t = list(map(int, t))
    # t = list(map(str, t))
    # amps = list(map(float, amps))
    # amps = list(map(str, amps))
    # plt.plot(t, amps, label=hemisphere)

xticks = []
for i in range(0, len(t), len(t) // 10):
    xticks.append(i)
ax.set_xticks(xticks)
ax.set_yticks([y_min, y_max])

plt.legend()
plt.show()