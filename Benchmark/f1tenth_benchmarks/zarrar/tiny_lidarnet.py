import time
import numpy as np
from numba import njit
from f1tenth_benchmarks.utils.BasePlanner import BasePlanner
import tensorflow as tf

class TinyLidarNet(BasePlanner):
    def __init__(self, test_id, skip_n, pre, model_path):
        super().__init__("TinyLidarNet", test_id)
        self.pre = pre
        self.skip_n = skip_n
        self.model_path = model_path
        self.name = 'TinyLidarNet'
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_details = self.interpreter.get_output_details()

    def linear_map(self, x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

    def render_waypoints(self, *args, **kwargs):
        pass
        
    def plan(self, obs):
        scans = obs['scan']
        noise = np.random.normal(0, 0.5, scans.shape)
        scans = scans + noise
        chunks = [scans[i:i+4] for i in range(0, len(scans), 4)]
        if self.pre == 1:
            scans = [np.mean(chunk) for chunk in chunks]
        elif self.pre == 2:
            scans = [np.max(chunk) for chunk in chunks]
        elif self.pre == 3:
            scans = [np.min(chunk) for chunk in chunks]
        
        else:
            scans = scans[::self.skip_n]

        if self.pre < 4:
            scans = np.array(scans)
            scans[scans>10] = 10
            scans = np.expand_dims(scans, axis=-1).astype(np.float32)
            scans = np.expand_dims(scans, axis=0)
            self.interpreter.set_tensor(self.input_index, scans)
            

            start_time = time.time()
            self.interpreter.invoke()
            inf_time = time.time() - start_time
            inf_time = inf_time*1000
            output = self.interpreter.get_tensor(self.output_details[0]['index'])

            steer = output[0,0]
            speed = output[0,1]
            min_speed = 1
            max_speed = 8
            #alpha = 0
            #new_max_speed = max_speed + alpha
            speed = self.linear_map(speed, 0, 1, min_speed, max_speed) #for all
            #speed = self.linear_map(speed, 0, 1, 2.5, 8) #for all
            # speed = self.linear_map(speed, 0, 1, 3, 8) #for moscow
            # speed = self.linear_map(speed, 0, 1, 1.0, 8) # max
            # speed = self.linear_map(speed, 0, 1, 1.0, 8) # mean
            # speed = self.linear_map(speed, 0, 1, 1, 8) #for min
            action = np.array([steer, speed])
        else:
            scans = np.array(scans)
            scans[scans>10] = 10
            # print(scans.shape)
            scans = np.asarray(scans).reshape(-1,1,1081,1).astype(np.float32)
            # print(scans.shape)
            self.interpreter.set_tensor(self.input_index, scans)
            

            start_time = time.time()
            self.interpreter.invoke()
            inf_time = time.time() - start_time
            inf_time = inf_time*1000
            output = self.interpreter.get_tensor(self.output_details[0]['index'])

            steer = output[0,0]
            speed = output[0,1]
            min_speed = 1
            max_speed = 5
            speed = self.linear_map(speed, 0, 1, min_speed, max_speed)
            action = np.array([steer, speed])

        return action