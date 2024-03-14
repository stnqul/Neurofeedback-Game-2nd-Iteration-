
# Stage 2: Preparing the EEG data for flicker-induced SSVEP detection
for j in range(2): # We only need the occipital lobe data for now
    ys_flicker[j] = mySensor.get_data()[j][-window_size:]

    # Method 1: Reducing the sample size from 250 to 60, i.e. to the flicker frequency
    # t = 0
    # it = 1
    # ys_flicker_reduced = []
    # while t < sample_len:
    #     curr_window = ys_flicker[j][t : min(t+4, sample_len)]
        
    #     # Correcting every 4 iterations for the 0.1(6) in 250/60 = 4.1(6)
    #     if it % 4 == 0 and sample_len - t > 4:
    #         curr_window.append(ys_flicker[j][t+4])
    #         t += 5
    #     else:
    #         t += 4
    #     ys_flicker_reduced.append(sum(curr_window) / len(curr_window))
    #     it += 1
    
    # ys_flicker[j] = ys_flicker_reduced

# Method 2: Using a moving window
# window_average = sum(ys_flicker) / len(ys_flicker)
# ys_flicker = [ys_flicker[t] - window_average for t in range(250)]

if len(ys_flicker[0]) == window_size:
    reg1_flicker.fit(np.array(xs_flicker).reshape(-1,1), np.array(ys_flicker[0]))
    reg2_flicker.fit(np.array(xs_flicker).reshape(-1,1), np.array(ys_flicker[1]))
    reg1_flicker_slope = reg1_flicker.coef_
    reg1_flicker_slope = reg1_flicker.coef_

    for t in range(window_size):
        ys1_flicker[0]