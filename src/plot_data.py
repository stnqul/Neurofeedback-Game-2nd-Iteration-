import sys
import numpy as np
import matplotlib.pyplot as plt

# SAMPLE_FREQ = 250 # Hz, the BrainBit's fequency
REFRESH_FREQ = 60 # Hz, the screen refresh rate

def to_freq_plot(amps, sample_freq):
    amps = np.array(amps)
    n = amps.size
    
    # print(f"n: {n}")
    # print(f"sample_freq: {sample_freq}")
    # print(f"1./sample_freq: {1./sample_freq}")
    
    freq = np.fft.rfftfreq(n, d=1./sample_freq)
    freq_amps = np.fft.rfft(amps)

    # Eliminating the 0th component (which corresp. to the sample frequency)
    freq = freq[1:]
    freq_amps = np.abs(freq_amps[1:])
    # print(f"First frequency bin:{freq[0]}")
    print(f"freq: {freq}")
    print(f"freq_amps: {freq_amps}")

    # The screen refresh frequency amplitudte is the greatest
    refresh_freq_idx = np.where(freq_amps == np.max(freq_amps))[0][0]

    # Eliminating the amplitudes corresponding to the screen refresh rate
    #   and to half of the sampling freqency (i.e. the last component)
    freqs_of_interes_amps = np.concatenate((freq_amps[:refresh_freq_idx], freq_amps[(refresh_freq_idx + 1):-1]), axis=0)
    print(f"Average of the frequencies of interest: {np.average(freqs_of_interes_amps)}")

    return freq, freq_amps

# Command parsing

if sys.argv[1] == "freq":
    FREQ_PLOT = True
elif sys.argv[1] == "amp":
    FREQ_PLOT = False
else:
    sys.exit("Command format is: python3 ./plot_data.py [freq/amp] [files_to_plot_from]")

# Plotting housekeeping
    
fig, ax = plt.subplots()
if FREQ_PLOT:
    ax.set_title("Frequency plot")
    ax.set_xlabel("Frequency bins")
    ax.set_ylabel("Frequency amplitude")
else:
    ax.set_title("Batch average")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude avg")
stem_colors = ['b', 'g']
curr_color_idx = 0

# File parsing and plotting

for file_path in sys.argv[2:]:
    
    if '\\' in file_path:
        hemisphere = file_path.split('\\')[-1].split('.')[0]
    else:
        hemisphere = file_path.split('/')[-1].split('.')[0]

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
        for i in range(len(split_lines[0])):
            amps_line = []
            for j in range(len(split_lines)):
                amps_line.append(split_lines[j][i])
            amps_avgs.append(np.average(amps_line))

    if FREQ_PLOT and sample_freq:
        freq, freq_amps = to_freq_plot(amps_avgs, sample_freq)
        plt.stem(freq, freq_amps, stem_colors[curr_color_idx], label=hemisphere)
        
        curr_color_idx += 1
        if curr_color_idx == len(stem_colors):
            curr_color_idx = 0
    else:
        x = range(len(amps_avgs))
        plt.plot(x, amps_avgs, label=hemisphere)

plt.legend()
plt.show()