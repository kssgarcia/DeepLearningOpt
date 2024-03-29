# %%
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import colors
import tensorflow as tf
from simp_solver.SIMP import optimization
import wandb

wandb.init(project='my-tensor')
print("Available GPUs:", tf.config.experimental.list_physical_devices('GPU'))

x1 = np.loadtxt('../simp/results_matlab/x_dataL.txt')
load_x1 = np.loadtxt('../simp/results_matlab/load_x_dataL.txt')
load_y1 = np.loadtxt('../simp/results_matlab/load_y_dataL.txt')
vol1 = np.loadtxt('../simp/results_matlab/vol_dataL.txt')
bc1 = np.loadtxt('../simp/results_matlab/bc_dataL.txt')

x2 = np.loadtxt('../simp/results_matlab/x_dataL2.txt')
load_x2 = np.loadtxt('../simp/results_matlab/load_x_dataL2.txt')
load_y2 = np.loadtxt('../simp/results_matlab/load_y_dataL2.txt')
vol2 = np.loadtxt('../simp/results_matlab/vol_dataL2.txt')
bc2 = np.loadtxt('../simp/results_matlab/bc_dataL2.txt')

x = np.concatenate((x1, x2), axis=1).T
load_x = np.concatenate((load_x1, load_x2), axis=1).T
load_y = np.concatenate((load_y1, load_y2), axis=1).T
vol = np.concatenate((vol1, vol2), axis=1).T
bc = np.concatenate((bc1, bc2), axis=1).T

input_shape = (61, 61)  # Input size of 61x61
num_channels = 4  # Number of channels in each input array
batch_size = bc.shape[0]  # Number of samples in each batch

input_data = np.zeros((batch_size,) + input_shape + (num_channels,))
for i in range(batch_size):
    input_data[i, :, :, 0] = bc[i].reshape((61,61))
    input_data[i, :, :, 1] = vol[i].reshape((61,61))
    input_data[i, :, :, 2] = load_x[i].reshape((61,61))
    input_data[i, :, :, 3] = load_y[i].reshape((61,61))
output_data = x.reshape((x.shape[0],60,60))

input_train = input_data[:-1000]
output_train = output_data[:-1000]

input_val = input_data[-1000:]
output_val = output_data[-1000:]

batch_size = input_train.shape[0]
# %%
# Import the model weights to wandb
artifact = wandb.use_artifact('model-weights:latest')
artifact_dir = artifact.download()

# %%
def pixel_accuracy(y_true, y_pred):
    y_true_binary = tf.round(y_true)
    y_pred_binary = tf.round(y_pred)
    pixel_accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true_binary, y_pred_binary), dtype=tf.float32))
    return pixel_accuracy

#model = tf.keras.models.load_model('../models/unn_merge_3', custom_objects={'pixel_accuracy': pixel_accuracy})
model = tf.keras.models.load_model('./test')
model.summary()

# %%
test_loss, test_accuracy = model.evaluate(input_val, output_val)

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

    bc = np.zeros((60+1, 60+1))
    bc[0, :] = 1

    vol = np.ones((60+1, 60+1)) * volfrac

    load_y = np.zeros((60+1, 60+1), dtype=int)
    load_y[40, -1] = l
    load_y[-1, 30] = l
    load_y[40, 0] = l
    load_x = np.zeros((60+1, 60+1), dtype=int)

    plt.figure()
    plt.matshow(load_y, cmap='gray')
    plt.show()

    new_input[0, :, :, 0] = bc
    new_input[0, :, :, 1] = vol
    new_input[0, :, :, 2] = load_x
    new_input[0, :, :, 3] = load_y
    
    return new_input 

#y = model.predict(custom_load(0.6, 1))
y = model.predict(input_val)

# %%
index = 400
plt.ion() 
fig,ax = plt.subplots(1,3)
ax[0].imshow(np.array(-y[index]).reshape(60, 60).T, cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[0].set_title('Predicted')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].matshow(-output_val[index].reshape(60, 60).T, cmap='gray')
#ax[1].imshow(-np.flipud(optimization(60, 1, 1, 61, 1, 30, 1, 0.6).reshape(60, 60)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[1].set_title('Expected')
ax[1].set_xticks([])
ax[1].set_yticks([])
# Hacer una grafica con la convolucion
ax[2].matshow(load_x[index].reshape(61, 61).T, cmap='gray')
ax[2].set_title('Load point')
ax[2].set_xticks([])
ax[2].set_yticks([])
fig.show()
