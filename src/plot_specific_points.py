import sys
import numpy as np
import matplotlib.pyplot as plt

def list_overlap_len(l1, l2):
    def list_overlap_len_rec(l1, l2, rel_last_idx):
        
        if rel_last_idx <= len(l2) - 1:
            l1_slice = l1[-(rel_last_idx + 1):]
            l2_slice = l2[:rel_last_idx + 1]
        elif rel_last_idx <= len(l1) - 1:
            l1_slice = l1[-(rel_last_idx + 1) : -(rel_last_idx + 1) + len(l2)]
            l2_slice = l2
        elif rel_last_idx <= len(l1) + len(l2) - 1:
            left_spill = rel_last_idx - len(l1) + 1
            slice_len = len(l2) - left_spill
            l1_slice = l1[:slice_len]
            l2_slice = l2[left_spill:]
        else:
            return None
        return len(l1_slice) if l1_slice == l2_slice else list_overlap_len_rec(l1, l2, rel_last_idx + 1)

    if len(l1) < len(l2):
        aux = l1
        l1 = l2
        l2 = aux
        
    return list_overlap_len_rec(l1,l2,0)

# Plotting code

fig, ax = plt.subplots()
ax.set_title("Hemisphere amplitudes graph")
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")

no_of_ranges = int(sys.argv[1])
point_range_length = int(sys.argv[2])

t = None
y_min = -1
y_max = -1

for file_path in sys.argv[3:]:
    if '\\' in file_path:
        hemisphere = file_path.split('\\')[-1].split('.')[0]
    else: # '/' in file_path
        hemisphere = file_path.split('/')[-1].split('.')[0]

    with open(file_path) as f:
        lines = f.readlines()
        if t is None:
            t = [line.split(' ')[1] for line in lines]
            # t = t[point_range_start : (point_range_start + point_range_length)]
        amps = [line.split(' ')[0] for line in lines]
        # print(amps)
        # amps = amps[point_range_start : (point_range_start + point_range_length)]

    if y_min == -1:
        y_min = min(amps)
    else:
        y_min = min(y_min, min(amps))
    if y_max == -1:
        y_max = max(amps)
    else:
        y_max = max(y_min, max(amps))

    t = range(point_range_length)
    for point_range_start in range(0, no_of_ranges * point_range_length, point_range_length):
        plt.plot(t,
                 amps[point_range_start : (point_range_start + point_range_length)],
                 label=hemisphere + f" {point_range_start}")
    
    batch_no = 1
    for point_range_start in range(0, len(amps), point_range_length):
        amps1 = amps[point_range_start : (point_range_start + point_range_length)]
        amps2 = amps[(point_range_start + point_range_length) : (point_range_start + 2 * point_range_length)]
        
        print(f"Batch no: {batch_no}")
        print(f"Overlap between batches {batch_no} {batch_no + 1}: {list_overlap_len(amps1, amps2)}")
        batch_no += 1

    # amps_avgs = [np.average(amps[i:i+point_range_length]) for i in range(0, len(amps), point_range_length)]
    # for i in range(len(amps_avgs) - 1):
    #     print(f"Diff between batches {i+1} and {i+2}: {np.abs(amps_avgs[i+1] - amps_avgs[i+2])}")

xticks = []
for i in range(0, point_range_length, point_range_length // 10):
    xticks.append(i)
ax.set_xticks(xticks)
ax.set_yticks([y_min, y_max])

# l1 = list(range(10))
# l2 = 5 * [0] + list(range(4))
# l3 = [1] * 10
# l4 = [7,8,9]
# print(list_overlap_len(l1,l4))

# plt.legend()
# plt.show()