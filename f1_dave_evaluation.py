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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import rosbag
import math
import os
import time

#========================================================
# Functions
#========================================================
def wait_for_key():
    input("Press any key to continue...")
    print("Continuing...")
def linear_map(x, x_min, x_max, y_min, y_max):
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min
wait_for_key()
#========================================================
#Get Data
#========================================================
#------------------------------------------------
#Training Data
#------------------------------------------------

lidar = []
servo = []
speed = []
max_speed = 0
temp_cnt = 1

for pth in ['qualifier_2/out.bag', 'f2.bag', 'f4.bag','test_data_nicholes.bag', 'test_data_nicholes_r.bag']:
    if(not os.path.exists(pth)):
        print(f"out.bag doesn't exists in {pth}")
        exit(0)
    good_bag = rosbag.Bag(pth)
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

lidar = np.asarray(lidar)
servo = np.asarray(servo)
speed = np.asarray(speed)
print(f'Loaded {len(lidar)} samples')

print(speed)
assert len(lidar) == len(servo) == len(speed)
print(f'Loaded {len(lidar)} samples')
print(f'Shape of Lidar: {lidar.shape}, Servo: {servo.shape}, Speed: {speed.shape}')

print(f'norm_speed: {np.linalg.norm(speed)}')
wait_for_key()


#------------------------------------------------
#Testing Data
#------------------------------------------------
print("---------------------\nTesting Data\n----------------------")
#speed = speed/norm_speed
t_lidar = []
t_servo = []
t_speed = []
t_max_speed = 0
t_temp_cnt = 1

for pth in ['test_data_nicholes.bag', 'test_data_nicholes_r.bag']:
    if(not os.path.exists(pth)):
        print(f"out.bag doesn't exists in {pth}")
        exit(0)
    good_bag = rosbag.Bag(pth)
    for topic, msg, t in good_bag.read_messages():
        if topic == 'Lidar':
            ranges = msg.ranges

            # Remove quandrant of LIDAR directly behind us
            t_lidar.append(ranges)

        if topic == 'Ackermann':
            data = msg.drive.steering_angle
            s_data = msg.drive.speed
            t_servo.append(data)

            if s_data>max_speed:
                max_speed = s_data
            s_data = linear_map(s_data, 0, 5, 0, 1)
            t_speed.append(s_data)

    print(f'Loaded {len(t_lidar)} samples')
    print(f'Shape of Lidar: {len(t_lidar)}, Servo: {len(t_servo)}, Speed: {len(t_speed)}')
    print(max(t_speed))
    wait_for_key()

t_lidar = np.asarray(t_lidar)
t_servo = np.asarray(t_servo)
t_speed = np.asarray(t_speed)
print(f'Loaded {len(t_lidar)} samples')

print(t_speed)
assert len(t_lidar) == len(t_servo) == len(t_speed)
print(f'Loaded {len(t_lidar)} samples')
print(f'Shape of Lidar: {t_lidar.shape}, Servo: {t_servo.shape}, Speed: {t_speed.shape}')
print(f'norm_speed: {np.linalg.norm(t_speed)}')
wait_for_key()
t_test_data = np.concatenate((t_servo[:, np.newaxis], t_speed[:, np.newaxis]), axis=1)
print(f'Test Data (servo,speed): {t_test_data.shape}')


#======================================================
# Split Dataset
#======================================================
print('Spliting Data to Train/Test')

test_data = np.concatenate((servo[:, np.newaxis], speed[:, np.newaxis]), axis=1)

print(f'Train Data(servo, speed): {test_data.shape}')

x_train, x_test, y_train, y_test = train_test_split(lidar, test_data, test_size = 0.15, shuffle=False)

#x_train, x_test, y_train, y_test = train_test_split(lidar, test_data, test_size = 0.15, shuffle=False)
#x_train, x_test, y_train, y_test = train_test_split(lidar, servo, test_size = 0.35)
print(f'Train Size: {len(x_train)}')
print(f'Train Size, y_train: {y_train.shape}')
print(f'Validation Size: {len(x_test)}')
print(f'y_test.shape{y_test.shape}')
wait_for_key()

#======================================================
# DNN Arch
#======================================================
num_lidar_range_values = len(lidar[0])
print(f'num_lidar_range_values: {num_lidar_range_values}')

# Mess around with strides, kernel_size, Max Pooling
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=24, kernel_size=10, strides=4, activation='relu', input_shape=(num_lidar_range_values, 1)),
    tf.keras.layers.Conv1D(filters=36, kernel_size=8, strides=4, activation='relu'),
    tf.keras.layers.Conv1D(filters=48, kernel_size=4, strides=2, activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(2, activation='tanh')
])
#======================================================
# Model Compilation
#======================================================
#lr = 3e-4
lr = 5e-5

optimizer = tf.keras.optimizers.Adam(lr)
model.compile(optimizer=optimizer, loss='huber')#, metrics = [r2]) 

#huber is noisy data else 'mean_squared_error'

print(model.summary())
wait_for_key()

#======================================================
# Model Fit
#======================================================
##See Data Balance in DeepPiCar

batch_size = 64
num_epochs = 20

start_time = time.time()
history = model.fit(lidar, test_data, epochs=num_epochs, batch_size=batch_size, validation_data=(x_test, y_test))
#history = model.fit(lidar, test_data, epochs=num_epochs, batch_size=batch_size, validation_data=(x_test, y_test))
#history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_test, y_test))

# Plot training and validation losses 
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

#plt.show()

#print(f'=============>{int(time.time() - start_time)} seconds<=============')

wait_for_key()


#======================================================
# Model Evaluation
#======================================================
print("==========================================")
print("Model Evaluation")
print("==========================================")


test_loss = model.evaluate(t_lidar, t_test_data)
#test_loss = model.evaluate(x_test, y_test)
print(f'test_loss = {test_loss}')

y_pred = model.predict(x_test)

#y_pred = model.predict(t_lidar)
#y_test = t_test_data
#print(f'y_pred.shape: {y_pred.shape}')
#accuracy = np.mean(y_pred == y_test)

r2 = r2_score(y_test, y_pred)
error = mean_squared_error(y_test, y_pred)

#print(f'Accuracy: {accuracy:.3f}%')
print('\nOverall Evaluation:')
print(f'r2: {r2:.3f}')
print(f'Error: {error}')

speed_test_loss = model.evaluate(t_lidar, t_test_data)
x_test = t_lidar
y_test = t_test_data
speed_y_pred = model.predict(x_test)[:, 1]  # Extracting the speed predictions

# Calculate R-squared (R2) score for Speed
speed_r2 = r2_score(y_test[:, 1], speed_y_pred)

# Calculate Mean Squared Error (MSE) for Speed
speed_error = mean_squared_error(y_test[:, 1], speed_y_pred)

print("\nSpeed Evaluation:")
print(f"Speed Test Loss: {speed_test_loss}")
print(f"Speed R-squared (R2) Score: {speed_r2}")
print(f"Speed Mean Squared Error (MSE): {speed_error}")

# Model Evaluation for Servo
servo_y_pred = model.predict(x_test)[:, 0]  # Extracting the servo predictions

# Calculate R-squared (R2) score for Servo
servo_r2 = r2_score(y_test[:, 0], servo_y_pred)

# Calculate Mean Squared Error (MSE) for Servo
servo_error = mean_squared_error(y_test[:, 0], servo_y_pred)

print("\nServo Evaluation:")
print(f"Servo R-squared (R2) Score: {servo_r2}")
print(f"Servo Mean Squared Error (MSE): {servo_error}\n")
wait_for_key()

#======================================================
# Prune
#======================================================
# Function to prune neurons
'''def prune_neurons(model, num_neurons_to_remove):
    # List to store new weights for each layer
    new_weights = []

    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            # Get the weights and biases of the current layer
            weights, biases = layer.get_weights()

            # Compute the importance of each neuron (e.g., L2 norm of weights)
            neuron_importance = np.linalg.norm(weights, axis=0)

            # Get the indices of the least important neurons
            least_important_neurons = np.argsort(neuron_importance)[:num_neurons_to_remove]

            # Remove the least important neurons by setting their weights to 0
            weights[:, least_important_neurons] = 0.0

            # Append the pruned weights and biases to the new_weights list
            new_weights.append([weights, biases])

            # Print the layer and indices of pruned neurons
            print(f'Layer {i + 1}: Pruned neurons indices: {least_important_neurons}')

    # Set the pruned weights in the model
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            # Check if the layer has trainable parameters (e.g., Dense layers)
            if i < len(new_weights):
                layer.set_weights(new_weights[i])
            else:
                print(f"Layer {i + 1} does not have trainable parameters.")



# Example usage
# Create a sample Keras model (replace with your actual model)
model = tf.keras.models.Model(inputs=model.inputs, outputs=model.outputs)

# Number of neurons to remove (change as needed)
num_neurons_to_remove = 10

# Prune the model
prune_neurons(model, num_neurons_to_remove)'''

def prune_neurons(model, num_neurons_to_remove):
    # List to store new weights for each layer
    new_weights = []

    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            # Get the weights and biases of the current layer
            weights, biases = layer.get_weights()

            # Compute the importance of each neuron (e.g., L2 norm of weights)
            neuron_importance = np.linalg.norm(weights, axis=0)

            # Get the indices of the least important neurons
            least_important_neurons = np.argsort(neuron_importance)[:num_neurons_to_remove]

            # Remove the least important neurons by setting their weights to 0
            weights[:, least_important_neurons] = 0.0

            # Append the pruned weights and biases to the new_weights list
            new_weights.append([weights, biases])

            # Print the layer and indices of pruned neurons
            print(f'Layer {i + 1}: Pruned neurons indices: {least_important_neurons}')

    # Set the pruned weights in the model
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            # Check if the layer has trainable parameters (e.g., Dense layers)
            if i < len(new_weights):
                layer.set_weights(new_weights[i])
            else:
                print(f"Layer {i + 1} does not have trainable parameters.")

    # Print the model summary after pruning
    print("Model Summary after Pruning:")
    model.summary()


# Example usage
# Create a sample Keras model (replace with your actual model)
# Assuming you have a model defined and trained before calling prune_neurons

# Number of neurons to remove (change as needed)
#num_neurons_to_remove = 10

# Prune the model
#prune_neurons(model, num_neurons_to_remove)
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude, PolynomialDecay

# Compute end step to finish pruning after 2 epochs.
batch_size = 64
epochs = 2
validation_split = 0.35  # 15% of the training set will be used for validation set.

num_images = len(lidar) * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
    'pruning_schedule': PolynomialDecay(initial_sparsity=0.50, final_sparsity=0.80, begin_step=0, end_step=end_step)
}

# Prune the model
pruned_model = prune_low_magnitude(model) #, **pruning_params)

# Print the model summary after pruning
pruned_model.summary()

# Compile the model
pruned_model.compile(optimizer='adam', loss='huber')


# Add the UpdatePruningStep callback
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep()
]

# Train the model
start_time = time.time()
history = pruned_model.fit(
    lidar,
    test_data,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
    callbacks=callbacks
)

# Save the pruned model
pruned_model.save('pruned_model.h5')
print("Pruned model saved.")


wait_for_key()

#======================================================
# Save Model
#======================================================
model_file = 'f1_tenth_model'
#model.save(model_file+'.h5')
print("Model Saved")


# Set TensorFlow logging level to debug
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.debugging.set_log_device_placement(True)


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.dump_graphviz_dir = './tflite_graph.dot'
# Enable verbose logging for the conversion process
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]


# Dynamic range quantization
# Reduced memory usage and faster computation 
# Statically quantizes only the weights from floating point to integer at conversion time, 
# which provides 8-bits of precision:
# Set verbose mode to True
#converter.verbose = True


quantized_tflite_model = converter.convert()

with open(model_file+".tflite", 'wb') as f:
    f.write(quantized_tflite_model)
    print (model_file+".tflite is saved. copy this file to the robot")
print('Tf_lite Model also saved')
wait_for_key()
#======================================================
# Evaluated TfLite Model
#======================================================

print("==========================================")
print("TFLite Evaluation")
print("==========================================")
model_name = './f1_tenth_model'
interpreter = tf.lite.Interpreter(model_path=model_name+'.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_details = interpreter.get_output_details()

output_lidar = lidar
output_servo = []
output_speed = []
y_test = test_data

hz = 100
period = 1.0 / hz


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

    # Print inference time information
    if dur > period:
        print("%.3f: took %.2f microseconds - deadline miss." % (ts - start, int(dur * 1000000)))
    #else:
    #    print("%.3f: took %.2f microseconds" % (ts - start, dur * 1000000))

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
plt.savefig('./graphs/inference_time_plot.png')
# Close the plot to free up resources
plt.close()
print(f"Plot saved")
#plt.show()

output_lidar = np.asarray(output_lidar)
output_servo = np.asarray(output_servo)
output_speed = np.asarray(output_speed)
assert len(output_lidar) == len(output_servo) == len(output_speed)
output = np.concatenate((output_servo[:, np.newaxis], output_speed[:, np.newaxis]), axis=1)

y_pred = output
print(f'y_pred.shape: {y_pred.shape}')
r2 = r2_score(y_test, y_pred)
error = mean_squared_error(y_test, y_pred)
print(f'r2: {r2:.3f}')
print(f'Error: {error}')


print('End')


import numpy as np

# Define the model
num_lidar_range_values = 1081  # Example input shape
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=24, kernel_size=10, strides=4, activation='relu', input_shape=(num_lidar_range_values, 1)),
    tf.keras.layers.Conv1D(filters=36, kernel_size=8, strides=4, activation='relu'),
    tf.keras.layers.Conv1D(filters=48, kernel_size=4, strides=2, activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(2, activation='tanh')
])

# Define the input shape for the first layer (excluding batch size)
input_shape = (num_lidar_range_values, 1)

# Function to calculate FLOPs for convolutional layers with strides
def calculate_conv_flops_with_strides(input_shape, kernel_shape, num_kernels, output_shape, strides):
    flops = 2 * (input_shape / strides) * kernel_shape * num_kernels * np.prod(output_shape)
    return flops

# Calculate FLOPs and MACs for each layer
output_shapes = [(None,) + layer.output_shape[1:] for layer in model.layers]
flops_list = []

for i in range(len(output_shapes)):
    layer_output_shape = output_shapes[i][1:]  # Exclude batch size from output shape
    if i < 5:  # Convolutional layers
        num_kernels = model.layers[i].get_weights()[0].shape[-1]
        kernel_shape = model.layers[i].get_weights()[0].shape[0]
        strides = model.layers[i].strides[0]
        flops = calculate_conv_flops_with_strides(input_shape[0], kernel_shape, num_kernels, layer_output_shape, strides)
    elif i >= 5 and i < 10:  # Fully connected layers
        input_size = np.prod(output_shapes[i-1][1:])
        output_size = np.prod(layer_output_shape)
        flops = 2 * input_size * output_size
    else:  # Pooling layers (ignoring strides for simplicity)
        flops = np.prod(layer_output_shape)
    flops_list.append(flops)

# Display FLOPs and MACs for each layer
for i, flops in enumerate(flops_list):
    print(f'Layer {i+1}: FLOPs = {flops:.2f}')

# Print model summary
model.summary()
