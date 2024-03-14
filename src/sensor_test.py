from itertools import count
from sensor import Sensor
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def reading_task(mySensor: Sensor):
    """
    Calls the sensor reading method from the given sensor object.
    """
    print('Start reading the sensor: ')
    mySensor.read_sensor_Ts(3)

if __name__ == "__main__":
    EEGSensor = Sensor()
    # readThread = Thread(target=reading_task, args=[EEGSensor])
    
    EEGSensor.activate_sensor()
    EEGSensor.print_sensor_information()
    
    x, y = [], []
    index = count()

    def animate(i):
        x.append(next(index))
        # y.append(random.randint(2,20))
        y.append(EEGSensor.read_sensor_Ts(100))
        print(y, "\n")

        plt.xlim(i-30,i+3)
        plt.style.use("ggplot")
        plt.plot(x,y, scalex=True, scaley=True, color="red")

    # TODO: Plot the EEG data live using matplotlib
    fig = plt.figure(figsize=(6,4))
    
    anim = FuncAnimation(fig, animate, interval=100)
    plt.show()

# EEG data animated plotting

# from matplotlib.animation import FuncAnimation
# matplotlib.use("agg")
# import matplotlib.backends.backend_agg as agg
# import random
# from itertools import count

# display_data = list(np.array(ys[3]))
# def animate(i):
#     # if self.x == []:
#     #     self.x = list(range(len(display_data)))
#     # else:
#     #     self.x += list(range(x[-1] + 1, x[-1] + len(display_data) + 1))
#     self.x += list(range(self.current_graph_times,
#                          self.current_graph_times + len(display_data)))
#     self.y += display_data
    
#     self.current_graph_times += len(self.x)

#     print(display_data)
#     plt.xlim(i-30,i+3)
#     plt.style.use("ggplot")
#     plt.plot(self.x,self.y, scalex=True, scaley=True, color="red")
    
# anim = FuncAnimation(self.fig, animate, interval=100)
# plt.show()

# time = np.array(range(graph_step * current_graph_iterations,
#                       graph_step * (current_graph_iterations + 1)))
# plt.xlim(time[0] - 30,
#          time[99] + 3)
# ax.plot(time, ys4_last)
# current_graph_iterations += 1

# fig.canvas.draw()
# canvas = fig.canvas
# renderer = canvas.get_renderer()
# raw_data = renderer.tostring_rgb()
# plt.close()

# size = canvas.get_width_height()

# surf = pygame.image.fromstring(raw_data, size, "RGB")
# self.screen.blit(surf, (graph_x, graph_y))
# pygame.display.flip()
    
# Odd/even averaging
# ys_occ_1_even = []
# ys_occ_1_odd = []
# ys_occ_2_even = []
# ys_occ_2_odd = []
# for i in range(0, window_size, 2):
#     ys_occ_1_even.append(ys_occ_1[i])
#     ys_occ_1_odd.append(ys_occ_1[i+1])
#     ys_occ_2_even.append(ys_occ_2[i])
#     ys_occ_2_odd.append(ys_occ_2[i+1])

# occ_1_delta = np.abs(np.average(ys_occ_1_even) - np.average(ys_occ_1_odd))
# occ_2_delta = np.abs(np.average(ys_occ_2_even) - np.average(ys_occ_2_odd))