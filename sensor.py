from time import sleep
import concurrent.futures
import threading
from neurosdk.scanner import Scanner
# from neurosdk.brainbit_sensor import BrainBitSensor
# from neurosdk.brainbit_black_sensor import BrainBitBlackSensor
from neurosdk.cmn_types import *
from datetime import datetime


class Sensor:
    def __init__(self):
        """
        Initialization of the sensor.
        """
        self.scanner = None
        self.sensor = None
        self.sensorFamily = None
        self.O1data = []
        self.O2data = []
        self.T3data = []
        self.T4data = []
        self.x_values = []
        self.resist_data = []
        self.threading_event = threading.Event()
        self.last_packnum = 0
        self.global_pack_counter = 0
        self.last_sec = datetime.now().second

    def activate_sensor(self):
        """
        Scans for devices via bluetooth, if a sensor (device) is found then assigns it to a sensor object in a separate thread using ThreadPool.
        """
        print("Scanning for devices for 5 sec...")
        try:
            def sensor_found(scanner, sensors):
                for index in range(len(sensors)):
                    print('Sensor found: %s' % sensors[index])

            def on_signal_data_received(sensor, data):
                self.O1data.append(data[0].O1)
                self.O2data.append(data[0].O2)
                self.T3data.append(data[0].T3)
                self.T4data.append(data[0].T4)
                
                # curr_packnum = data[0].PackNum
                # self.global_pack_counter += 1
                # curr_sec = datetime.now().second
                # print(f"curr_sec: {curr_sec}")
                # if(curr_sec > self.last_sec):
                #     print(f"Packs spanned: {self.global_pack_counter - self.last_packnum}")
                #     self.last_packnum = self.global_pack_counter
                #     self.last_sec = curr_sec
                #     print("\n")

            def on_resist_data_received(sensor, data):
                self.resist_data.append(data)
                print(type(data))

            # Scanning for devices:
            self.scanner = Scanner([SensorFamily.LEBrainBit]) # Sensor name may change due to further updates

            self.scanner.sensorsChanged = sensor_found
            self.scanner.start()
            sleep(5)
            self.scanner.stop()
            # self.scanner.sensorsChanged = None

            # Getting the sensor information from the found device:
            sensorsInfo = self.scanner.sensors()
            current_sensor_info = sensorsInfo[0]

            def device_connection(sensor_info):
                return self.scanner.create_sensor(sensor_info)
            
            # Starting the sensor as a new thread:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(device_connection, current_sensor_info)
                self.sensor = future.result()
                print("Device connected")
            # Defining the sensorFamily:
            self.sensorFamily = self.sensor.sens_family
            # Assigning the callback function into sensor:
            if self.sensor.is_supported_feature(SensorFeature.Signal): # FeatureSignal used to be here
                self.sensor.signalDataReceived = on_signal_data_received
            # if self.sensor.is_supported_feature(SensorFeature.Resist):
                # self.sensor.resistDataReceived = on_resist_data_received

        except Exception as err:
            print(err)
        
    def deactivate_sensor(self):
        """
        Disconnects from the sensor (device). Terminates the device scanner and the previously constructed sensor object.
        """
        if self.sensor != None:
            print("Disconnected from sensor")
            del self.sensor
        if self.scanner != None:
            print("Removed scanner")
            del self.scanner

    def read_sensor_1s(self):
        """
        Runs the callback function for 1 second.
        """
        if self.sensor.is_supported_command(SensorCommand.StartSignal): # CommandStartSignal was here before
            self.sensor.exec_command(SensorCommand.StartSignal)
            print("Started reading signal...")
            sleep(1)
            self.sensor.exec_command(SensorCommand.StopSignal) # CommandStopSignal was here before
            print("Stopped reading signal...")
        else:
            print("A problem with sensor or command occured during reading sensor.")
        
        # if self.sensor.is_supported_command(SensorCommand.StartResist):
        #     self.sensor.exec_command(SensorCommand.StartResist)
        #     print("Started reading the resistance...")
        #     sleep(1)
        #     self.sensor.exec_command(SensorCommand.StopResist)
        #     print("Stopped reading the resistance...")
        # else:
        #     print("A problem with sensor or command occured while reading sensor resistance.")

    def read_sensor_Ts(self, T):
        """
        Runs the callback function for T seconds.
        """
        if self.sensor.is_supported_command(SensorCommand.StartSignal): # CommandStartSignal was here before
            self.sensor.exec_command(SensorCommand.StartSignal)
            print("Started reading signal...")
            self.threading_event.wait(timeout=T)
            self.sensor.exec_command(SensorCommand.StopSignal) # CommandStopSignal was here before
            print("Stopped reading signal...")
        else:
            print("A problem with sensor or command occured while reading the sensor signal.")
        
        if self.sensor.is_supported_command(SensorCommand.StartResist):
            self.sensor.exec_command(SensorCommand.StartResist)
            print("Started reading the resistance...")
            self.threading_event.wait(timeout=T)
            self.sensor.exec_command(SensorCommand.StopResist)
            print("Stopped reading the resistance...")
        else:
            print("A problem with sensor or command occured while reading sensor resistance.")

    def print_sensor_information(self):
        """
        Prints the sensor (device) information. (e.g. features, commands, sampling_frequency).
        """
        print("Sensor Family information:")
        print(self.sensorFamily)
        print("Sensor information:")
        print(self.sensor.features)
        print(self.sensor.commands)
        print(self.sensor.parameters)
        print(self.sensor.name)
        print(self.sensor.state)
        print(self.sensor.address)
        print(self.sensor.serial_number)
        print(self.sensor.batt_power)
        print(self.sensor.sampling_frequency)
        print(self.sensor.gain)
        print(self.sensor.data_offset)
        print(self.sensor.version)

    def get_data(self):
        """
        Getter function for the received data (all 4 channels).

        :return: data (a list of 4-tuples)
        """
        return (self.O1data, self.O2data, self.T3data, self.T4data)