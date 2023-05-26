import cv2
import os
import time
import numpy as np
import sys
import math
import rospy
import message_filters
import tensorflow as tf
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped


lid = '/scan_filtered' #'/scan_filtered' #'/scan'
joy = '/vesc/joy'
prev = 0
curr = 0

is_joy = rospy.get_param("/is_joy")
print(f'is_joy: {is_joy}')
#===================================================
def callback(l):
    global lidar_data
    ldata = l.ranges
    #eighth = int(len(ldata)/8)
    #ldata = np.array(ldata[eighth:-eighth]).astype(np.float32)
    ldata = np.expand_dims(ldata, axis=-1).astype(np.float32)
    ldata = np.expand_dims(ldata, axis=0)
    lidar_data = ldata

def button_callback(j):
    global prev
    global curr
    global is_joy
    curr = j.buttons[0]
    if(curr == 1 and curr!=prev):
        print(f'X Pressed')
        rospy.set_param('/is_joy', not is_joy)
        is_joy = rospy.get_param("/is_joy")
    prev = curr

#===================================================
def load_model():
    global interpreter
    global input_index
    global output_details
    global model

    print("Model")    
    model_name = './f1_tenth_model'
    model = tf.keras.models.load_model(model_name+'.h5')
    interpreter = tf.lite.Interpreter(model_path=model_name+'.tflite')
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_details = interpreter.get_output_details()#[0]["index"]

def dnn_output():
    global lidar_data
    if lidar_data is None:
        return 0.
    ##lidar_data = np.expand_dims(lidar_data).astype(np.float32)
    interpreter.set_tensor(input_index,lidar_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    #servo = output[0]
    servo = output[0,0]
    speed = output[0,1]
    #print(f'Servo: {servo}, Speed: {speed}')
    print(f'Servo: {servo}')
    return servo, speed
    #return servo

def linear_map(x, x_min, x_max, y_min, y_max):
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min
def undo_min_max_scale(x, x_min, x_max):
    return x * (x_max - x_min) + x_min
#===================================================
rospy.init_node('Autonomous')
servo_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, queue_size=10)

rospy.Subscriber(joy, Joy, button_callback)
rospy.Subscriber(lid, LaserScan, callback)
hz = 80
rate = rospy.Rate(hz)
period = 1.0/hz

start_ts = time.time()
load_model()
while not rospy.is_shutdown():
    is_joy = rospy.get_param('/is_joy')
    print('Manual Control: ON')
    
    if not is_joy:
        #print('Vroommmmmm......')
        ts = time.time()
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        
        servo, speed = dnn_output()
        #servo = dnn_output()
        speed = linear_map(speed, 0, 1, -0.8, 7)#-0.6 to 7
        #speed = 0.4 + (0.55 * lidar_data[0][lidar_data.shape[1]//2][0])
        print(lidar_data[len(lidar_data)//2])
        print(lidar_data.shape)
        print(f'speed: {speed}')
        msg.drive.speed = speed 
        #undo_min_max_scale(speed, 0, 5.0) #linear_map(speed, 0, 1,0.5 ,5)
        #speed = 1 
        msg.drive.steering_angle = servo
        
        #if abs(servo) <= 0.1:
        #    speed = 1

        #msg.drive.speed = speed


        dur = time.time() - ts
        if dur > period:
            print("%.3f: took %d ms - deadline miss."% (ts - start_ts, int(dur * 1000)))
        else:
            print("%.3f: took %d ms"    % (ts - start_ts, int(dur * 1000)))

        servo_pub.publish(msg)
        #print(msg)
    rate.sleep()
