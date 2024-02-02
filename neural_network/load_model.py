# %%
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import colors
import tensorflow as tf
from simp_solver.SIMP import optimization

# Create dummy input data
x1 = np.loadtxt('../simp/results_matlab/x_dataL.txt')
load_x1 = np.loadtxt('../simp/results_matlab/load_x_dataL.txt')
load_y1 = np.loadtxt('../simp/results_matlab/load_y_dataL.txt')
bc1 = np.loadtxt('../simp/results_matlab/bc_dataL.txt')

x2 = np.loadtxt('../simp/results_matlab/x_dataL2.txt')
load_x2 = np.loadtxt('../simp/results_matlab/load_x_dataL2.txt')
load_y2 = np.loadtxt('../simp/results_matlab/load_y_dataL2.txt')
bc2 = np.loadtxt('../simp/results_matlab/bc_dataL2.txt')

x = np.concatenate((x1, x2), axis=1).T
load_x = np.concatenate((load_x1, load_x2), axis=1).T
load_y = np.concatenate((load_y1, load_y2), axis=1).T
bc = np.concatenate((bc1, bc2), axis=1).T

print(x.shape)
print(load_x.shape)
print(load_y.shape)
print(bc.shape)

# Generate random input data
input_shape = (61, 61)  # Input size of 61x61
num_channels = 3  # Number of channels in each input array
batch_size = bc.shape[0]  # Number of samples in each batch

input_data = np.zeros((batch_size,) + input_shape + (num_channels,))
for i in range(batch_size):
    input_data[i, :, :, 0] = bc[i].reshape((61,61))
    input_data[i, :, :, 1] = load_x[i].reshape((61,61))
    input_data[i, :, :, 1] = load_y[i].reshape((61,61))

output_data = x.reshape((x.shape[0],60,60))

input_train = input_data[:-1000]
output_train = output_data[:-1000]

input_test = input_data[-1000:]
output_test = output_data[-1000:]

batch_size = input_train.shape[0]

# %%
def pixel_accuracy(y_true, y_pred):
    y_true_binary = tf.round(y_true)
    y_pred_binary = tf.round(y_pred)
    pixel_accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true_binary, y_pred_binary), dtype=tf.float32))
    return pixel_accuracy

#model = tf.keras.models.load_model('../models/unn_merge_3', custom_objects={'pixel_accuracy': pixel_accuracy})
model = tf.keras.models.load_model('../models/vit_matlab')
model.summary()

# %%
test_loss, test_accuracy = model.evaluate(input_test, output_test)

# %%
y_pred = model.predict(input_train)[:,:,:,0]
y_true = output_train

y_pred_binary = tf.round(y_pred)
y_true_binary = tf.round(y_true)

# Convert to boolean tensors for comparison
y_true_bool = tf.cast(y_true_binary, dtype=tf.bool)
y_pred_bool = tf.cast(y_pred_binary, dtype=tf.bool)

# Calculate pixel-wise accuracy for each image
pixel_accuracy_per_image = tf.reduce_mean(tf.cast(tf.equal(y_true_bool, y_pred_bool), dtype=tf.float32), axis=(1, 2))

# Calculate global accuracy for the entire batch
global_accuracy = tf.reduce_mean(pixel_accuracy_per_image)

# Get the result as a numpy array
global_accuracy_result = global_accuracy.numpy()

print(global_accuracy_result)

# %%
# Calculate pixel-wise accuracy for each image
pixel_accuracy_per_image = np.mean(np.equal(y_true, y_pred), axis=(1, 2))

# Calculate global accuracy for the entire batch
global_accuracy = np.mean(pixel_accuracy_per_image)

# Get the result as a scalar
print("Global Pixel Accuracy:", global_accuracy)

# %%
def custom_load(volfrac, l):
    new_input = np.zeros((1,) + (61,61) + (num_channels,))
    bc = np.ones((60+1, 60+1)) *  volfrac
    bc[:, 0] = 1
    load = np.zeros((60+1, 60+1), dtype=int)
    load[-1, -1] = l
    load[-61, -1] = l
    load[-30, -1] = l

    new_input[0, :, :, 0] = bc
    new_input[0, :, :, 1] = load
    
    return new_input 

input_mod = np.concatenate((input_train, custom_load(0.6, 1)), axis=0)

y = model.predict(input_mod)


# %%

index = 3000
plt.ion() 
fig,ax = plt.subplots(1,3)
ax[0].imshow(np.flipud(np.array(-y[index]).reshape(60, 60)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
#ax[0].imshow(np.array(y_custom).reshape(60, 60), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[0].set_title('Predicted')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].matshow(-np.flipud(output_train[index].reshape(60, 60)), cmap='gray')
#ax[1].imshow(-np.flipud(optimization(60, 1, 1, 61, 1, 30, 1, 0.6).reshape(60, 60)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[1].set_title('Expected')
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[2].matshow(load[index].reshape(61, 61), cmap='gray')
ax[2].set_title('Load point')
ax[2].set_xticks([])
ax[2].set_yticks([])
fig.show()
