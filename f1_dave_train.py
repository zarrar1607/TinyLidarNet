#Requirement Library
#import cv2
#import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rosbag
import math
import os
import time

#========================================================
# Functions
def wait_for_key():
    input("Press any key to continue...")
    print("Continuing...")
def linear_map(x, x_min, x_max, y_min, y_max):
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min
wait_for_key()
#========================================================
#Get Data
data_path = 'Dataset/out.bag'
if(not os.path.exists(data_path)):
    print(f"out.bag doesn't exists in {data_path}")

#Preprocess
bag = rosbag.Bag(data_path)
lidar = []
servo = []
speed = []
max_speed = 0
temp_cnt = 1

good_bag = rosbag.Bag('Dataset/qualifier_out.bag')
for topic, msg, t in good_bag.read_messages():
    if topic == 'Lidar':
        ranges = msg.ranges

        # Remove quandrant of LIDAR directly behind us
        lidar.append(ranges)

    if topic == 'Ackermann':
        data = msg.drive.steering_angle
        s_data = msg.drive.speed
        servo.append(data)
        if s_data>max_speed:
            max_speed = s_data
        s_data = linear_map(s_data, 0, 5, 0, 1)
        speed.append(s_data)

print(f'Loaded {len(lidar)} samples')
print(f'Shape of Lidar: {len(lidar)}, Servo: {len(servo)}, Speed: {len(speed)}')
print(max(speed))
wait_for_key()

f2 = rosbag.Bag('Dataset/f2.bag')
for topic, msg, t in f2.read_messages():
    if topic == 'Lidar':
        
        ranges = msg.ranges
        lidar.append(ranges)
        
    if topic == 'Ackermann':
        data = msg.drive.steering_angle
        s_data = msg.drive.speed
        servo.append(data)
        if s_data>max_speed:
            max_speed = s_data
        s_data = linear_map(s_data, 0, 5, 0, 1)
        speed.append(s_data)

print(f'Loaded {len(lidar)} samples')
print(f'Shape of Lidar: {len(lidar)}, Servo: {len(servo)}, Speed: {len(speed)}')
print(max(speed))
wait_for_key()


f4 = rosbag.Bag('Dataset/f4.bag')
for topic, msg, t in f4.read_messages():
    if topic == 'Lidar':

        ranges = msg.ranges
        lidar.append(ranges)

    if topic == 'Ackermann':
        data = msg.drive.steering_angle
        s_data = msg.drive.speed
        servo.append(data)
        if s_data>max_speed:
            max_speed = s_data
        s_data = linear_map(s_data, 0, 5, 0, 1)
        speed.append(s_data)

print(f'Loaded {len(lidar)} samples')
print(f'Shape of Lidar: {len(lidar)}, Servo: {len(servo)}, Speed: {len(speed)}')
print(max_speed)
wait_for_key()


lidar = np.asarray(lidar)
servo = np.asarray(servo)
speed = np.asarray(speed)
print(f'Loaded {len(lidar)} samples')

#norm_speed = np.linalg.norm(speed)
#speed = speed/norm_speed
print(speed)
assert len(lidar) == len(servo) == len(speed)
print(f'Loaded {len(lidar)} samples')
print(f'Shape of Lidar: {lidar.shape}, Servo: {servo.shape}, Speed: {speed.shape}')

print(f'norm_speed: {np.linalg.norm(speed)}')
wait_for_key()

#======================================================
# Split Dataset
print('Spliting Data to Train/Test')

test_data = np.concatenate((servo[:, np.newaxis], speed[:, np.newaxis]), axis=1)
print(f'Test Data {test_data.shape}')

x_train, x_test, y_train, y_test = train_test_split(lidar, test_data, test_size = 0.35)
#x_train, x_test, y_train, y_test = train_test_split(lidar, servo, test_size = 0.35)
print(f'Train Size: {len(x_train)}')
print(f'Test Size: {len(x_test)}')
print(f'y_test.shape{y_test.shape}')
wait_for_key()

#======================================================
# DNN Arch
num_lidar_range_values = len(lidar[0])
print(f'num_lidar_range_values: {num_lidar_range_values}')

# Mess around with strides, kernel_size, Max Pooling
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=24, kernel_size=10, strides=4, activation='relu', input_shape=(num_lidar_range_values, 1)),
    tf.keras.layers.Conv1D(filters=36, kernel_size=8, strides=4, activation='relu'),
    tf.keras.layers.Conv1D(filters=48, kernel_size=4, strides=2, activation='relu'),
    #tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    #tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    #tf.keras.layers.Dense(2, activation='tanh')
    tf.keras.layers.Dense(2, activation='tanh')
])
#print(model.summary())


#======================================================
# Model Compilation
#lr = 3e-4
lr = 5e-5

optimizer = tf.keras.optimizers.Adam(lr)
model.compile(optimizer=optimizer, loss='huber')#, metrics = [r2]) #huber is noisy data else 'mean_squared_error'

print(model.summary())
wait_for_key()

#======================================================
# Model Fit
##See Data Balance in DeepPiCar

batch_size = 64
num_epochs = 20

start_time = time.time()
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_test, y_test))

# Plot training and validation losses 
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print(f'=============>{int(time.time() - start_time)} seconds<=============')

wait_for_key()


#======================================================
# Model Evaluation
test_loss = model.evaluate(x_test, y_test)
print(f'test_loss = {test_loss}')

y_pred = model.predict(x_test)
print(f'y_pred.shape: {y_pred.shape}')
accuracy = np.mean(y_pred == y_test)
r2 = r2_score(y_test, y_pred)
error = mean_squared_error(y_test, y_pred)
print(f'Accuracy: {accuracy:.3f}')
print(f'r2: {r2:.3f}')
print(f'Error: {error}')

#======================================================
# Save Model
model_file = 'f1_tenth_model'
model.save(model_file+'.h5')
print("Model Saved")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()

with open(model_file+".tflite", 'wb') as f:
    f.write(quantized_tflite_model)
    print (model_file+".tflite is saved. copy this file to the robot")
print('Tf_lite Model also saved')

#End
print('End')

