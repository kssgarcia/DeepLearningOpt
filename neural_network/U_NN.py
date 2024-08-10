# %%
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import tensorflow as tf
from tensorflow import keras
from models import UNN_model
import pandas as pd

print(tf.config.list_physical_devices('GPU'))

x1 = np.loadtxt('../simp/results_matlab/x_dataL.txt')
load_x1 = np.loadtxt('../simp/results_matlab/load_x_dataL.txt')
load_y1 = np.loadtxt('../simp/results_matlab/load_y_dataL.txt')
vol1 = np.loadtxt('../simp/results_matlab/vol_dataL.txt')
bc1 = np.loadtxt('../simp/results_matlab/bc_dataL.txt')

x = x1.T
load_x = load_x1.T
load_y = load_y1.T
vol = vol1.T
bc = bc1.T

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

input_train = input_data[:500]
output_train = output_data[:500]

input_val = input_data[-100:]
output_val = output_data[-100:]

batch_size = input_train.shape[0]

input_shape = (61, 61, num_channels)

print(input_train.shape)
print(output_train.shape)

# Normalize
output_val = np.where(output_val > 0.5, 1, 0)
output_train = np.where(output_train > 0.5, 1, 0)

if np.any((output_val > 0) & (output_val < 1)):
    print("output_val has elements between 0 and 1")
else:
    print("output_val does not have elements between 0 and 1")

test_n = 1

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    f"./best_unn_grokking_{test_n}/cp.ckpt",
    monitor="loss",
    mode="min",
    save_best_only=True,
    save_weights_only=True,
    verbose= 1,
)

#adam_optimizer = keras.optimizers.Adam(learning_rate=1e-4)

decay_steps = (input_train.shape[0] // 32)*5
print(decay_steps)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=decay_steps,
        decay_rate=0.86,  
        staircase=True  
        )
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model = UNN_model((61,61,num_channels))
model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.MeanAbsoluteError()])
history = model.fit(input_train, output_train, epochs=8000, batch_size=32, validation_data=[input_val, output_val], callbacks=[checkpoint_callback])
#history = model.fit(input_val, output_val, epochs=200, batch_size=10, validation_split=0.2, callbacks=[checkpoint_callback])

model.save(f"./model_unn_grokking_{test_n}")

y = model.predict(input_val)

index = 40
plt.ion() 
fig,ax = plt.subplots(1,2)
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
plt.savefig(f"plots/unn_result_grokking_{test_n}.png")  # Save the plot as an image
fig.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"plots/loss_unn_grokking_{test_n}.png")  # Save the plot as an image
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f"plots/accuracy_unn_grokking_{test_n}.png")  # Save the plot as an image

# Save training and validation loss to a CSV file
loss_data = {
    'Training Loss': history.history['loss'],
    'Validation Loss': history.history['val_loss'],
    'Training Accuracy': history.history['accuracy'],
    'Validation Accuracy': history.history['val_accuracy']
}
loss_df = pd.DataFrame(loss_data)
loss_df.to_csv(f"./unn_loss_data_grokking_{test_n}.csv", index=False)
