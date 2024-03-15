#Requirement Library
#import cv2
#import sklearn
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rosbag
import math
import time
import subprocess



gpu_available = tf.test.is_gpu_available() #cross check to see any GPU usage
print('GPU AVAILABLE', gpu_available)

#========================================================
# Functions
#========================================================
def wait_for_key():
    input("Press any key to continue...")
    print("Continuing...")
def linear_map(x, x_min, x_max, y_min, y_max):
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min
def huber_loss(y_true, y_pred, delta=1.0):
    # Calculate the absolute error
    error = np.abs(y_true - y_pred)

    # Calculate the loss based on the Huber loss formula
    loss = np.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))

    # Calculate the mean loss
    mean_loss = np.mean(loss)

    return mean_loss

#========================================================
#Get Data
#========================================================
total_number_samples = 0
lidar = []
servo = []
speed = []
test_lidar = []
test_servo = []
test_speed = []
max_speed = 0
temp_cnt = 1
max_speed = 0
min_speed = 0

for pth in [
        #'../../lab1.bag',
        '../../qualifier_2/out.bag', 
        '../../f2.bag', 
        '../../f4.bag',
        #'../../test_data_nicholes.bag', 
        #'../test_data_nicholes_r.bag'
        ]:
    if(not os.path.exists(pth)):
        print(f"out.bag doesn't exists in {pth}")
        exit(0)
    good_bag = rosbag.Bag(pth)

    lidar_data = []
    servo_data = []
    speed_data = []

    for topic, msg, t in good_bag.read_messages():
        if topic == 'Lidar':
            ranges = msg.ranges#[180:900]
            #chunks = [ranges[i:i+4] for i in range(0, len(ranges), 4)]
            #ranges = [np.mean(chunk) for chunk in chunks]
            #ranges = [np.min(chunk) for chunk in chunks]
            #ranges = [np.max(chunk) for chunk in chunks]
            total_number_samples+=1
            lidar_data.append(ranges)

            #lidar_data.append(ranges[::-1])
        if topic == 'Ackermann':
            data = msg.drive.steering_angle
            s_data = msg.drive.speed
            
            servo_data.append(data)
            #servo_data.append(data)
            if s_data>max_speed:
                max_speed = s_data
            #s_data = linear_map(s_data, 0, 5, 0, 1)
            speed_data.append(s_data)
            #speed_data.append(s_data)

    lidar_data = np.array(lidar_data) 
    servo_data = np.array(servo_data)
    speed_data = np.array(speed_data)
    if max_speed < np.max(speed_data):
        max_speed = np.max(speed_data)
    if min_speed > np.min(speed_data):
        min_speed = np.min(speed_data)
    data = np.concatenate((servo_data[:, np.newaxis], speed_data[:, np.newaxis]), axis=1)

    from sklearn.utils import shuffle
    shuffled_data = shuffle(data, random_state = 62)#62
    shuffled_lidar_data = shuffle(lidar_data, random_state = 62)#62
    # Check the shape of shuffled_data to ensure it has two columns
    print("Shape of shuffled_data:", shuffled_data.shape)

    # Split the shuffled data into train and test sets
    train_ratio = 0.85  # Adjust the ratio as needed
    train_samples = int(train_ratio * len(shuffled_lidar_data))
    x_train_bag, x_test_bag = shuffled_lidar_data[:train_samples], shuffled_lidar_data[train_samples:]

    # Check the shape of x_train_bag and x_test_bag
    print("Shape of x_train_bag:", x_train_bag.shape)
    print("Shape of x_test_bag:", x_test_bag.shape)

    # Extract servo and speed values from y_train_bag and y_test_bag
    y_train_bag = shuffled_data[:train_samples]
    y_test_bag = shuffled_data[train_samples:]

    # Check the shape of y_train_bag and y_test_bag
    print("Shape of y_train_bag:", y_train_bag.shape)
    print("Shape of y_test_bag:", y_test_bag.shape)

    # Extend the appropriate lists
    lidar.extend(x_train_bag)
    servo.extend(y_train_bag[:, 0])  # Extract servo values
    speed.extend(y_train_bag[:, 1])  # Extract speed values

    test_lidar.extend(x_test_bag)
    test_servo.extend(y_test_bag[:, 0])  # Extract servo values for test
    test_speed.extend(y_test_bag[:, 1])  # Extract speed values for test
    print(f'\nData in {pth}:')
    print(f'Shape of Train Data --- Lidar: {len(lidar)}, Servo: {len(servo)}, Speed: {len(speed)}')
    print(f'Shape of Test Data --- Lidar: {len(test_lidar)}, servo: {len(test_servo)}, Speed: {len(test_speed)}')

    wait_for_key()

print(f'Overall Samples = {total_number_samples}')
lidar = np.asarray(lidar)
servo = np.asarray(servo)
speed = np.asarray(speed)
speed = linear_map(speed, min_speed, max_speed, 0, 1)
print(f'Loaded {len(lidar)} Training samples ---- {(len(lidar)/total_number_samples)*100:0.2f}% of overall')
test_lidar = np.asarray(test_lidar)
test_servo = np.asarray(test_servo)
test_speed = np.asarray(test_speed)
test_speed = linear_map(test_speed, min_speed, max_speed, 0, 1)

print(f'Min_speed: {min_speed}')
print(f'Max_speed: {max_speed}')
print(f'Loaded {len(test_lidar)} Testing samples ---- {(len(test_lidar)/total_number_samples)*100:0.2f}% of overall\n')

assert len(lidar) == len(servo) == len(speed)
assert len(test_lidar) == len(test_servo) == len(test_speed)
         
wait_for_key()

#======================================================
# Split Dataset
#======================================================
print('Spliting Data to Train/Test')

train_data = np.concatenate((servo[:, np.newaxis], speed[:, np.newaxis]), axis=1)
test_data =  np.concatenate((test_servo[:, np.newaxis], test_speed[:, np.newaxis]), axis=1)
#validation_data = np.concatenate((validation_servo[:, np.newaxis], validation_speed[:, np.newaxis]), axis=1)

print(f'Train Data(lidar): {lidar.shape}')
print(f'Train Data(servo,speed): {train_data.shape}')
#print(f'Validation Data(servo, speed): {validation_data.shape}')
print(f'Test Data(lidar): {test_lidar.shape}')
print(f'Test Data(servo, speed): {test_data.shape}')


#x_train, x_test, y_train, y_test = train_test_split(lidar, train_data, test_size = 0.15, shuffle=False)
#x_train, x_test, y_train, y_test = train_test_split(lidar, train_data, test_size = 0.15,  stratify = train_data)

#print(f'Train Size: {len(x_train)}')
#print(f'Train Size, y_train: {y_train.shape}')
#print(f'Validation Size: {len(x_test)}')
#print(f'y_test.shape{y_test.shape}\n')
wait_for_key()

#======================================================
# DNN Arch
#======================================================
num_lidar_range_values = len(lidar[0])
print(f'num_lidar_range_values: {num_lidar_range_values}')

# Mess around with strides, kernel_size, Max Pooling
model = tf.keras.Sequential([
    #tf.keras.layers.Conv1D(filters=24, kernel_size=10, strides=4, activation='relu', input_shape=(num_lidar_range_values, 1)),
    #tf.keras.layers.Conv1D(filters=36, kernel_size=8, strides=4, activation='relu'),
    #tf.keras.layers.Conv1D(filters=48, kernel_size=4, strides=2, activation='relu'),
    #tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    #tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    #tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(100, activation='relu'),
    #tf.keras.layers.Dense(50, activation='relu'),
    #tf.keras.layers.Dense(10, activation='relu'),
    #tf.keras.layers.Dense(2, activation='tanh')

    #Dropouts
    tf.keras.layers.Conv1D(filters=24, kernel_size=10, strides=4, activation='relu', input_shape=(num_lidar_range_values, 1)),
    tf.keras.layers.Conv1D(filters=36, kernel_size=8, strides=4, activation='relu'),
    tf.keras.layers.Conv1D(filters=48, kernel_size=4, strides=2, activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='tanh')


    #Unifying
    #tf.keras.layers.Flatten(input_shape=(num_lidar_range_values, 1)),
    #tf.keras.layers.Dense(100, activation='relu'),
    #tf.keras.layers.Dense(100, activation='relu'),
    #tf.keras.layers.Dense(2, activation='tanh')

    #MLP 256
    #tf.keras.layers.Flatten(input_shape=(num_lidar_range_values, 1)),
    #tf.keras.layers.Dense(256, activation='relu'),
    #tf.keras.layers.Dense(256, activation='relu'),
    #tf.keras.layers.Dense(2, activation='tanh')

    #MLP 128
    #tf.keras.layers.Flatten(input_shape=(num_lidar_range_values, 1)),
    #tf.keras.layers.Dense(128, activation='relu'),
    #tf.keras.layers.Dense(128, activation='relu'),
    #tf.keras.layers.Dense(2, activation='tanh')

    #C2 C3 C4
    #tf.keras.layers.Conv1D(filters=24, kernel_size=10, strides=4, activation='relu', input_shape=(num_lidar_range_values, 1)),
    #tf.keras.layers.Conv1D(filters=36, kernel_size=8, strides=4, activation='relu'),
    #tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(100, activation='relu'),
    #tf.keras.layers.Dense(50, activation='relu'),
    #tf.keras.layers.Dense(10, activation='relu'),
    #tf.keras.layers.Dense(2, activation='tanh')

    #C1 C3
    #tf.keras.layers.Conv1D(filters=24, kernel_size=10, strides=4, activation='relu', input_shape=(num_lidar_range_values, 1)),
    #tf.keras.layers.Conv1D(filters=48, kernel_size=4, strides=2, activation='relu'),
    #tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    #tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(100, activation='relu'),
    #tf.keras.layers.Dense(50, activation='relu'),
    #tf.keras.layers.Dense(10, activation='relu'),
    #tf.keras.layers.Dense(2, activation='tanh')
    
    #D0 D2
    #tf.keras.layers.Conv1D(filters=24, kernel_size=10, strides=4, activation='relu', input_shape=(num_lidar_range_values, 1)),
    #tf.keras.layers.Conv1D(filters=36, kernel_size=8, strides=4, activation='relu'),
    #tf.keras.layers.Conv1D(filters=48, kernel_size=4, strides=2, activation='relu'),
    #tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    #tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    #tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(50, activation='relu'),
    #tf.keras.layers.Dense(2, activation='tanh')
    
    #2xK
    #tf.keras.layers.Conv1D(filters=24, kernel_size=20, strides=4, activation='relu', input_shape=(num_lidar_range_values, 1)),
    #tf.keras.layers.Conv1D(filters=36, kernel_size=16, strides=4, activation='relu'),
    #tf.keras.layers.Conv1D(filters=48, kernel_size=8, strides=2, activation='relu'),
    #tf.keras.layers.Conv1D(filters=64, kernel_size=6, activation='relu'),
    #tf.keras.layers.Conv1D(filters=64, kernel_size=6, activation='relu'),
    #tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(100, activation='relu'),
    #tf.keras.layers.Dense(50, activation='relu'),
    #tf.keras.layers.Dense(10, activation='relu'),
    #tf.keras.layers.Dense(2, activation='tanh')
    
    #2xS
    #tf.keras.layers.Conv1D(filters=24, kernel_size=10, strides=5, activation='relu', input_shape=(num_lidar_range_values, 1)),
    #tf.keras.layers.Conv1D(filters=36, kernel_size=8, strides=4, activation='relu'),
    #tf.keras.layers.Conv1D(filters=48, kernel_size=4, strides=3, activation='relu'),
    #tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'),
    #tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'),
    #tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(100, activation='relu'),
    #tf.keras.layers.Dense(50, activation='relu'),
    #tf.keras.layers.Dense(10, activation='relu'),
    #tf.keras.layers.Dense(2, activation='tanh')

])

#======================================================
# Model Compilation
#======================================================
#lr = 3e-4
lr = 5e-5
#lr = 0.001
#lr = 0.00042

optimizer = tf.keras.optimizers.Adam(lr)
model.compile(optimizer=optimizer, loss='huber')#, metrics = 'mean_squared_error') 

#huber is noisy data else 'mean_squared_error'

print(model.summary())
wait_for_key()

#======================================================
# Model Fit
#======================================================
##See Data Balance in DeepPiCar

batch_size = 64
num_epochs = 20

batch_size = 128
num_epochs = 100


start_time = time.time()
history = model.fit(lidar, train_data, epochs=num_epochs, batch_size=batch_size, validation_data=(test_lidar, test_data))#, shuffle=False)
#history = model.fit({"lidar":lidar,"acc":speed}, train_data, epochs=num_epochs, batch_size=batch_size, validation_data=({"lidar":test_lidar,"acc":test_speed}, test_data))#, shuffle=False)

print(f'=============>{int(time.time() - start_time)} seconds<=============')


# Plot training and validation losses 
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./loss_curve.png')
#plt.show()
plt.close()

wait_for_key()


#======================================================
# Model Evaluation
#======================================================
print("==========================================")
print("Model Evaluation")
print("==========================================")


test_loss = model.evaluate(test_lidar, test_data)
#test_loss = model.evaluate({"lidar": test_lidar, "acc": test_speed}, test_data)

print(f'test_loss = {test_loss}')

y_pred = model.predict(test_lidar)
r2 = r2_score(test_data, y_pred)
error = mean_squared_error(test_data, y_pred)

print('\nOverall Evaluation:')
print(f'r2: {r2:.3f}')
print(f'Error: {error}')

speed_test_loss = model.evaluate(test_lidar, test_data)
x_test = test_lidar
y_test = test_data
speed_y_pred = model.predict(x_test)[:, 1]  # Extracting the speed predictions
# Calculate R-squared (R2) score for Speed
speed_r2 = r2_score(y_test[:, 1], speed_y_pred)
# Calculate Mean Squared Error (MSE) for Speed
speed_error = mean_squared_error(y_test[:, 1], speed_y_pred)
print("\nSpeed Evaluation:")
print(f"Speed Test Loss: {huber_loss(y_test[:,1], speed_y_pred)}")
print(f"Speed R-squared (R2) Score: {speed_r2}")
print(f"Speed Mean Squared Error (MSE): {speed_error}")


# Model Evaluation for Servo
servo_y_pred = model.predict(x_test)[:, 0]  # Extracting the servo predictions
# Calculate R-squared (R2) score for Servo
servo_r2 = r2_score(y_test[:, 0], servo_y_pred)
# Calculate Mean Squared Error (MSE) for Servo
servo_error = mean_squared_error(y_test[:, 0], servo_y_pred)
print("\nServo Evaluation:")
print(f"Servo Test Loss: {huber_loss(y_test[:,0],servo_y_pred)}")
print(f"Servo R-squared (R2) Score: {servo_r2}")
print(f"Servo Mean Squared Error (MSE): {servo_error}\n")
wait_for_key()
#======================================================
# Prune
#======================================================

#======================================================
# Save Model
#======================================================
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model_file = 'f1_tenth_model_diff_dropout'
#model.save(model_file+'.h5')
print("Model Saved")

#Non Qunatized
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_model_path = model_file+"_noquantized.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
#output_header_file = tflite_model_path[:-7]+'.cc'
#subprocess.run(["xxd", "-i", tflite_model_path], stdout=open(output_header_file, "w"))
#print("No Quantized C header file generated successfully:", output_header_file)




rep_32 = lidar.astype(np.float32)
rep_32 = np.expand_dims(rep_32, -1)
# Convert the NumPy array to a TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices(rep_32)
# Define the representative_data_gen function
def representative_data_gen():
    for input_value in dataset.batch(len(lidar)).take(rep_32.shape[0]):
        # Print information about each input value
        #print(f'Type: {type(input_value)}, Shape: {input_value.shape}')
        # Yield the tensor directly
        yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

quantized_tflite_model = converter.convert()
tflite_model_path = model_file+"_int8.tflite"
with open(model_file+"_int8.tflite", 'wb') as f:
    f.write(quantized_tflite_model)
    print (model_file+"_int8.tflite is saved. copy this file to the robot")
print('Tf_lite Micro Model also saved')
#output_header_file = tflite_model_path[:-7]+'.cc'
#subprocess.run(["xxd", "-i", tflite_model_path], stdout=open(output_header_file, "w"))
#print("int8 C header file generated successfully:", output_header_file)


print('Tf_lite Model also saved')
wait_for_key()
#======================================================
# Evaluated TfLite Model
#=====================================================_
print("==========================================")
print("TFLite Evaluation")
print("==========================================")

model_name = './f1_tenth_model_diff_dropout_noquantized'
interpreter = tf.lite.Interpreter(model_path=model_name+'.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_details = interpreter.get_output_details()

print(test_lidar.shape)
output_lidar = test_lidar
output_servo = []
output_speed = []
y_test = test_data

hz = 80
period = 1.0 / hz


# Initialize a list to store inference times in microseconds
inference_times_micros = []


# Iterate through the lidar data
for lidar_data in output_lidar:
    # Preprocess lidar data for inference
    lidar_data = np.expand_dims(lidar_data, axis=-1).astype(np.float32)
    lidar_data = np.expand_dims(lidar_data, axis=0)

    # Check for empty lidar data
    if lidar_data is None:
        continue

    # Measure inference time
    ts = time.time()
    interpreter.set_tensor(input_index, lidar_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    dur = time.time() - ts

    # Convert inference time to microseconds
    inference_time_micros = dur * 1e6
    inference_times_micros.append(inference_time_micros)

    # Print inference time information
    if dur > period:
        print("%.3f: took %.2f microseconds - deadline miss." % (dur, int(dur * 1000000)))
    #else:
        #print("%.3f: took %.2f microseconds" % (ts - start, dur * 1000000))

    # Extract servo and speed output from the model
    servo = output[0, 0]
    speed = output[0, 1]

    # Append output servo and speed
    output_servo.append(servo)
    output_speed.append(speed)

# Calculate average and maximum inference times in microseconds
average_inference_time_micros = np.mean(inference_times_micros)
max_inference_time_micros = np.max(inference_times_micros)

# Print inference time statistics
print("Average Inference Time: %.2f microseconds" % average_inference_time_micros)
print("Maximum Inference Time: %.2f microseconds" % max_inference_time_micros)

# Plot inference times
arr = np.array(inference_times_micros)
perc99 = np.percentile(arr, 99)
arr = arr[arr < perc99]
plt.plot(arr)
plt.xlabel('Inference Iteration')
plt.ylabel('Inference Time (microseconds)')
plt.title('Inference Time per Iteration')

# Save the plot as an image
#plt.savefig('./graphs/inference_time_plot.png')
# Close the plot to free up resources
#plt.close()
#print(f"Plot saved")
#plt.show()

output_lidar = np.asarray(output_lidar)
output_servo = np.asarray(output_servo)
output_speed = np.asarray(output_speed)
assert len(output_lidar) == len(output_servo) == len(output_speed)
output = np.concatenate((output_servo[:, np.newaxis], output_speed[:, np.newaxis]), axis=1)

y_pred = output
#print(f'y_pred.shape: {y_pred.shape}')
r2 = r2_score(test_data, y_pred)
error = mean_squared_error(test_data, y_pred)
#print(f'r2: {r2:.3f}')
#print(f'Error: {error}')

print('End')

model_files = [
    #f1_tenth_model_diff_noquantized.tflite',
    #'f1_tenth_model.tflite',
    #'f1_tenth_modelfloat16.tflite',
    'f1_tenth_model_diff_2xS_int8.tflite'
]

# Initialize empty lists to store results for each model
all_inference_times_micros = []
all_errors = []
all_r2_scores = []

for model_name in model_files:
    interpreter = tf.lite.Interpreter(model_path=model_name)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_details = interpreter.get_output_details()

    output_lidar = test_lidar
    output_servo = []
    output_speed = []

    # Initialize a list to store inference times in microseconds
    inference_times_micros = []
    start = time.time()

    # Iterate through the lidar data
    for lidar_data in output_lidar:
        # Preprocess lidar data for inference
        lidar_data = np.expand_dims(lidar_data, axis=-1).astype(np.float32)
        lidar_data = np.expand_dims(lidar_data, axis=0)

        # Check for empty lidar data
        if lidar_data is None:
            continue

        # Measure inference time
        ts = time.time()
        interpreter.set_tensor(input_index, lidar_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        dur = time.time() - ts

        # Convert inference time to microseconds
        inference_time_micros = dur * 1e6
        inference_times_micros.append(inference_time_micros)
       
        #print(f"GPU: {tf.config.list_physical_devices('GPU')}")
        #print(f"CPU: {tf.config.list_physical_devices('CPU')}")

        # Print inference time information
        if dur > period:
            print("%.3f: took %.2f microseconds - deadline miss." % (ts - start, int(dur * 1000000)))

        # Extract servo and speed output from the model
        servo = output[0, 0]
        speed = output[0, 1]

        # Append output servo and speed
        output_servo.append(servo)
        output_speed.append(speed)

    output_lidar = np.asarray(output_lidar)
    output_servo = np.asarray(output_servo)
    output_speed = np.asarray(output_speed)
    assert len(output_lidar) == len(output_servo) == len(output_speed)
    output = np.concatenate((output_servo[:, np.newaxis], output_speed[:, np.newaxis]), axis=1)
    y_pred = output

    # Calculate average and maximum inference times in microseconds
    arr = np.array(inference_times_micros)
    perc99 = np.percentile(arr,99)
    arr = arr[arr < perc99]
    average_inference_time_micros = np.mean(arr)
    max_inference_time_micros = np.max(arr)

    # Append results for this model
    all_inference_times_micros.append(inference_times_micros)
    all_errors.append(mean_squared_error(test_data, y_pred))
    all_r2_scores.append(r2_score(test_data, y_pred))
    
    loss = model.evaluate(test_lidar, test_data)

    y_pred = output
    # Print inference time statistics for this model
    print(f"Model: {model_name}")
    print("Average Inference Time: %.2f microseconds" % average_inference_time_micros)
    print("Maximum Inference Time: %.2f microseconds" % max_inference_time_micros)
    print("Standard Deviation: %.2f microseconds" % np.std(inference_times_micros))
    print(f'R2 Score: {r2_score(test_data, y_pred)}')
    print(f'MSE: {mean_squared_error(test_data, y_pred)}')
    print(f'Huber Loss: {huber_loss(test_data,y_pred)}\n')	
'''
# Process results from all models
for i, model_name in enumerate(model_files):
    print(f"Results for Model: {model_name}")
    print(f"Average Inference Time: {np.mean(all_inference_times_micros[i]):.2f} microseconds")
    print(f"Maximum Inference Time: {np.max(all_inference_times_micros[i]):.2f} microseconds")
    print(f"Standard Deviation: {np.std(all_inference_times_micros[i]):.2f} microseconds")
    print(f'R2 Score: {all_r2_scores[i]:.3f}')
    print(f'Error: {all_errors[i]}')
    print()
	

    # Plot inference times for this model
    plt.plot(all_inference_times_micros[i], label=model_name)
    plt.xlabel('Inference Iteration')
    plt.ylabel('Inference Time (microseconds)')
    plt.title('Inference Time per Iteration')
    plt.ylim(0)
    plt.legend()
    plt.savefig(f'./graphs/inference_time_plot_{model_name}.png')
    plt.close()


    arr = np.array(all_inference_times_micros[i])
    perc99 = np.percentile(arr,99)
    arr = arr[arr < perc99]
    plt.plot(arr, label=model_name)
    plt.xlabel('Inference Iteration')
    plt.ylabel('Inference Time (microseconds)')
    plt.title('Inference Time per Iteration')
    plt.ylim(0)
    plt.legend()
    plt.savefig(f'./graphs/99_perct_inference_time_plot_{model_name}.png')
    plt.close()



'''
# Initialize an empty string to store the statistics
statistics_text = ""

for i, model_name in enumerate(model_files):
    # Process results for this model
    statistics_text += f"Results for Model: {model_name}\n"
    statistics_text += f"Average Inference Time: {np.mean(all_inference_times_micros[i]):.2f} microseconds\n"
    statistics_text += f"Maximum Inference Time: {np.max(all_inference_times_micros[i]):.2f} microseconds\n"
    statistics_text += f"Standard Deviation: {np.std(all_inference_times_micros[i]):.2f} microseconds\n"
    statistics_text += f'R2 Score: {all_r2_scores[i]:.3f}\n'
    statistics_text += f'Error: {all_errors[i]}\n\n'

    # Print inference time statistics for this model
    #print(f"Model: {model_name}")
    #print("Average Inference Time: %.2f microseconds" % average_inference_time_micros)
    #print("Maximum Inference Time: %.2f microseconds" % max_inference_time_micros)
    #print(f'R2 Score: {r2_score(test_data, y_pred)}')
    #print(f'MSE: {mean_squared_error(test_data, y_pred)}\n')

# Save the statistics to a text file
'''
with open("inference_time_statistics.txt", "w") as text_file:
    text_file.write(statistics_text)

print("Inference time statistics saved to 'inference_time_statistics.txt'")


# Plot inference times for all models
for i, model_name in enumerate(model_files):
    plt.plot(all_inference_times_micros[i], label=model_name)

plt.ylim(0)
plt.xlabel('Inference Iteration')
plt.ylabel('Inference Time (microseconds)')
plt.title('Inference Time per Iteration')
plt.legend()

# Save the plot as an image
plt.savefig('./graphs/inference_time_plot_all_models.png')

plt.close()

# Plot inference times for all models
for i, model_name in enumerate(model_files):
    arr = np.array(all_inference_times_micros[i])
    perc99 = np.percentile(arr,99)
    arr = arr[arr < perc99]
    plt.plot(arr, label=model_name)

plt.ylim(0)
plt.xlabel('Inference Iteration')
plt.ylabel('Inference Time (microseconds)')
plt.title('Inference Time per Iteration')
plt.legend()

# Save the plot as an image
plt.savefig('./graphs/99_perct_inference_time_plot_all_models.png')

'''
print(f"Plot saved")

# Close the plot to free up resources
plt.close()
