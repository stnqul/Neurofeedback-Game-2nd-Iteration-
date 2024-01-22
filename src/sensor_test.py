from itertools import count
from sensor import Sensor
from time import sleep
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import random
# import concurrent.futures
# from threading import *
# from neurosdk.scanner import Scanner
# from neurosdk.brainbit_sensor import BrainBitSensor
# from neurosdk.cmn_types import *

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