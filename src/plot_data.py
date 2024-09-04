import sys
import numpy as np
import matplotlib.pyplot as plt

# SAMPLE_FREQ = 250 # Hz, the BrainBit's fequency
REFRESH_FREQ = 60 # Hz, the screen refresh rate

def to_freq_plot(amps, sample_freq, hemisphere):
    amps = np.array(amps)
    n = amps.size
    
    # Debugging
    # print(f"n: {n}")
    # print(f"sample_freq: {sample_freq}")
    # print(f"1./sample_freq: {1./sample_freq}")
    
    freq = np.fft.rfftfreq(n, d=1./sample_freq)
    freq_amps = np.fft.rfft(amps)

    freq_full = freq
    freq_amps_full = np.abs(freq_amps)

    print(f"freq_full: {freq_full}")
    print(f"freq_amps_full: {freq_amps_full}")

    # Eliminating the 0th and the last component (which correspond to the sample frequency and to half of it, respectively)
    freq = freq[1:-1]
    freq_amps = np.abs(freq_amps[1:-1])

    # Debugging
    # print(f"First frequency bin:{freq[0]}")
    # print(f"Frequency bins: {freq}")
    # print(f"freq_amps: {freq_amps}")
    
    # Eliminating the component corresponding to the monitor refresh rate (i.e. the one closest to 60Hz)
    freq_refresh_rate_diff = np.abs(freq - REFRESH_FREQ)
    refresh_rate_idx = np.where(freq_refresh_rate_diff == np.min(freq_refresh_rate_diff))[0][0]
    freq_of_interest_amps = np.concatenate((freq_amps[:refresh_rate_idx], freq_amps[(refresh_rate_idx + 1):]), axis=0)
    freq_of_interest_amps_avg = np.average(freq_of_interest_amps)
    
    # Debugging
    # print(f"refresh_rate_idx: {refresh_rate_idx}")
    # print(f"freq: {freq}")
    # print(f"freq_full: {freq_full}")
    # print(f"freq_amps: {freq_amps}")
    # print(f"freq_amps_full: {freq_amps_full}")
    # print(f"freq_of_interest_amps: {freq_of_interest_amps}")

    print(f"Average of the frequencies of interest for {hemisphere}: {freq_of_interest_amps_avg:.2e}\n")

    return freq, freq_amps, freq_of_interest_amps_avg, freq_full, freq_amps_full


# Command parsing

args = sys.argv[1:]
no_args_min = 3
next_arg_idx = 0
error_msg = "Command format is: python3 ./plot_data.py [plot type: freq/amp] <save> <plot_name> [files_to_plot_from]"

if len(args) < no_args_min:
    sys.exit(error_msg)

if args[next_arg_idx] == 'freq':
    FREQ_PLOT = True
elif args[next_arg_idx] == 'amp':
    FREQ_PLOT = False
else:
    sys.exit(error_msg)

next_arg_idx += 1

if args[next_arg_idx] == 'save':
    SAVE_PLOT = True
    path_to_save = None
    next_arg_idx += 1
else:
    SAVE_PLOT = False

if '/' in args[next_arg_idx] or '\\' in args[next_arg_idx]: # no plot name
    PLOT_NAME = None
else:
    split_arg = args[next_arg_idx].split('_')
    split_arg[0] = split_arg[0].capitalize()
    PLOT_NAME = ' '.join(split_arg)
    next_arg_idx += 1

# if sys.argv[2] == 'yes':
#     SAVE_PLOT = True
# elif sys.argv[2] == 'no':
#     SAVE_PLOT = False
# else:
#     sys.exit(error_msg)

# Plot housekeeping

fig, ax = plt.subplots()
if FREQ_PLOT:
    if not PLOT_NAME:
        PLOT_NAME = "Frequency plot"
    ax.set_xlabel("Frequency bins")
    ax.set_ylabel("Frequency amplitude")
else:
    if not PLOT_NAME:
        PLOT_NAME = "Batch average"
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude avg")

ax.set_title(PLOT_NAME)

stem_colors = ['b', 'g', 'r', 'y']
curr_color_idx = 0

freq_of_interest_avgs = []

# File parsing and plotting

for file_path in args[next_arg_idx:]:
    if '\\' in file_path:
        path_splitter = '\\'
    else:
        path_splitter = '/'
    split_path = file_path.split(path_splitter) 

    file_name = split_path[-1].split('.')[0]
    if SAVE_PLOT and not path_to_save:
        path_to_save = '/'.join(split_path[:-2] + [PLOT_NAME + '.png'])

    lobe, no, spec = file_name.split('_')[:3]
    if lobe == 'occ':
        lobe = 'occipital'
    else:
        lobe = 'temporal'
    
    if no == '1':
        side = 'Left'
    else:
        side = 'Right'
    
    if spec == 'basic':
        spec = '(flicker)'
    elif spec == 'no':
        spec = '(no flicker)'
    else: # spec == 'left' or spec == 'right'
        spec = split_path[-2].split(' ')[-1]

    # elif spec == 'left':
    #     spec = '(right flicker)'
    # else:
    #     spec = '(left flicker)'

    hemisphere = ' '.join([side, lobe, spec])

    with open(file_path) as f:
        lines = f.readlines()
        split_lines = list(map(lambda l: list(filter(lambda e: e != '\n', l)),
                            map(lambda l: l.split(' '), lines)))
        split_lines = list(map(lambda l: list(map(lambda e: float(e), l)), split_lines))

        # print(f"len(split_lines[0]): {len(split_lines[0])}")
        
        sample_freq = None
        if len(split_lines[-1]) == 1:
            sample_freq = split_lines[-1][0]
            split_lines = split_lines[:-1]

        amps_avgs = []
        split_lines_row_len = min(len(split_lines[0]), len(split_lines[1]))
        for i in range(split_lines_row_len):
            amps_line = []
            for j in range(len(split_lines)):
                amps_line.append(split_lines[j][i])
            amps_avgs.append(np.average(amps_line))

    if FREQ_PLOT and sample_freq:
        freq, freq_amps, freq_of_interest_amps, freq_full, freq_amps_full = to_freq_plot(amps_avgs, sample_freq, hemisphere)
        
        plt.stem(freq, freq_amps, stem_colors[curr_color_idx], label=hemisphere)

        curr_color_idx += 1
        if curr_color_idx == len(stem_colors):
            curr_color_idx = 0

        freq_of_interest_avgs.append(freq_of_interest_amps)
    else:
        x = range(len(amps_avgs))
        plt.plot(x, amps_avgs, label=hemisphere)

if len(freq_of_interest_avgs) == 2:
    print(f"Diff btw avgs: {freq_of_interest_avgs[0] - freq_of_interest_avgs[1] :.2e}\n")

plt.legend()
plt.savefig(path_to_save) if SAVE_PLOT else None
plt.show()