# %%
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import keras
from keras import layers, regularizers

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

input_train = input_data[-1000:]
output_train = output_data[-1000:]

input_val = input_data[-100:]
output_val = output_data[-100:]

batch_size = input_train.shape[0]

input_shape = (61, 61, num_channels)

# Normalize
output_val = np.where(output_val > 0.5, 1, 0)
output_train = np.where(output_train > 0.5, 1, 0)

if np.any((output_val > 0) & (output_val < 1)):
    print("output_val has elements between 0 and 1")
else:
    print("output_val does not have elements between 0 and 1")

output_train = np.expand_dims(output_train, axis=-1)
output_val = np.expand_dims(output_val, axis=-1)

# %%
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        H = patches.shape[1]
        patches = tf.reshape(patches, [-1, H*H, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )(self.positions)

    def call(self, patch):
        encoded = self.projection(patch) + self.position_embedding
        return encoded

class MLP(layers.Layer):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.dense_layers = [layers.Dense(units, activation=tf.nn.gelu) for units in self.hidden_units]
        self.dropout_layers = [layers.Dropout(self.dropout_rate) for _ in self.hidden_units]

    def call(self, inputs):
        x = inputs
        for dense, dropout in zip(self.dense_layers, self.dropout_layers):
            x = dense(x)
            x = dropout(x)
        return x

class DecodingBlock(layers.Layer):
    def __init__(self, filters, strides, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides= strides
        self.trainable = trainable
        self.conv2DTranspose = layers.Conv2DTranspose(self.filters, (self.strides, self.strides), strides=(self.strides, self.strides), activation='relu', padding='same', trainable=self.trainable)
        self.batchnorm1 = layers.BatchNormalization()
        self.conv2D = layers.Conv2D(self.filters, (3, 3), activation='relu', padding='same', trainable=self.trainable)
        self.batchnorm2 = layers.BatchNormalization()

    def call(self, input_layer):
        x = self.conv2DTranspose(input_layer)
        x = self.batchnorm1(x)
        x = self.conv2D(x)
        x = self.batchnorm2(x)
        return x

class DecodingBlockSkip(layers.Layer):
    def __init__(self, filters, stride, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.concat = layers.Concatenate(axis=-1)
        self.conv2DTranspose = layers.Conv2DTranspose(self.filters, (self.stride, self.stride), strides=(self.stride, self.stride), activation='relu', padding='same')
        self.batchnorm1 = layers.BatchNormalization()
        self.conv2D = layers.Conv2D(self.filters, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.05))
        self.batchnorm2 = layers.BatchNormalization()
        self.dropout = layers.Dropout(rate=0.1) # New line

    def call(self, input_layer, concat_layer):
        x = self.concat([input_layer, concat_layer])
        x = self.conv2DTranspose(x)
        x = self.dropout(x) # New line
        x = self.batchnorm1(x)
        x = self.conv2D(x)
        x = self.batchnorm2(x)
        return x

class EncodingBlockSkip(layers.Layer):
    def __init__(self, filters, stride, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.conv1 = layers.Conv2D(self.filters, (3, 3), activation='relu', padding='same')
        self.batchnorm1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(self.filters, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.05))
        self.batchnorm2 = layers.BatchNormalization()
        self.dropout = layers.Dropout(rate=0.1) # New line
        if self.stride > 1:
            self.pool = layers.MaxPooling2D((self.stride, self.stride))
        else:
            self.batchnorm3 = layers.BatchNormalization()

    def call(self, input_layer):
        x = self.conv1(input_layer)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.dropout(x) # New line
        x = self.batchnorm2(x)
        if self.stride > 1:
            x = self.pool(x)
        else:
            x = self.batchnorm3(x)
        return x

class TransformerBlock(layers.Layer):
    def __init__(self, projection_dim, num_heads, transformer_units, transformer_layers, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.transformer_units = transformer_units
        self.transformer_layers = transformer_layers
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
        )
        self.mlp = MLP(hidden_units=self.transformer_units, dropout_rate=0.1)
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.add1 = layers.Add()
        self.add2 = layers.Add()

    def call(self, inputs):
        x = inputs
        for _ in range(self.transformer_layers):
            # Layer normalization 1.
            x1 = self.layer_norm1(x)
            # Create a multi-head attention layer.
            attention_output = self.attention(x1, x1)
            # Skip connection 1.
            x2 = self.add1([attention_output, x])
            # Layer normalization 2.
            x3 = self.layer_norm2(x2)
            # MLP.
            x3 = self.mlp(x3)
            # Skip connection 2.
            x = self.add2([x3, x2])

        return self.layer_norm3(x)

class HybridModel(keras.Model):
    def __init__(self, patch_size, projection_dim, num_heads, transformer_units, transformer_layers, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.transformer_units = transformer_units
        self.transformer_layers = transformer_layers
        self.initial = keras.Sequential([
            layers.Conv2D(32, kernel_size=(2, 2), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization()
        ])
        self.encoded1 = EncodingBlockSkip(64, 2)
        self.encoded2 = EncodingBlockSkip(128, 2)
        self.encoded3 = EncodingBlockSkip(256, 3)
        self.patches = Patches(self.patch_size)
        self.patchencoder = PatchEncoder(25, self.projection_dim)
        self.transformerblock = TransformerBlock(self.projection_dim, self.num_heads, self.transformer_units, self.transformer_layers)
        self.decoded1 = DecodingBlockSkip(256, 3)
        self.decoded2 = DecodingBlockSkip(128, 2)
        self.decoded3 = DecodingBlockSkip(64, 2)
        self.decoded4 = DecodingBlockSkip(32, 1)
        self.last = keras.Sequential([
            layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(16, kernel_size=(1, 1), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(16, kernel_size=(1, 1), activation='relu', padding='same'),
            layers.BatchNormalization(),
        ])
        self.output_tensor = layers.Conv2D(1, (1, 1), activation='sigmoid')

    def call(self, inputs):
        initial = self.initial(inputs)
        encoded1 = self.encoded1(initial)
        encoded2 = self.encoded2(encoded1)
        encoded3 = self.encoded3(encoded2)
        
        self.num_patches = (encoded2.shape[2] // self.patch_size) ** 2

        x = self.patches(encoded2)
        x = self.patchencoder(x)

        x = self.transformerblock(x)

        reshape_dim = int(np.sqrt(x.shape[1]))
        x = tf.reshape(x, [-1, reshape_dim, reshape_dim, self.projection_dim])

        x = self.decoded1(x, encoded3)
        x = self.decoded2(x, encoded2)
        x = self.decoded3(x, encoded1)
        #x = self.decoded4(x, initial)

        x = self.last(x)

        return self.output_tensor(x)

    def build_graph(self, input_shape):
        """
        ref: https://www.kaggle.com/code/ipythonx/tf-hybrid-efficientnet-swin-transformer-gradcam
        """
        x = keras.Input(shape=input_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))

    def summary(self, input_shape=(61, 61, 4)):
        return self.build_graph(input_shape).summary()

test_n = 17

patch_size = 3  
projection_dim = 128
num_heads = 8
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 4
input_shape = (61,61,4)

model = HybridModel(patch_size, projection_dim, num_heads, transformer_units, transformer_layers)
model.summary()

def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Binary cross-entropy loss
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    # # SSIM loss
    # ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    # Edge-aware loss
    y_true_edges = tf.image.sobel_edges(y_true)
    y_pred_edges = tf.image.sobel_edges(y_pred)
    edge_loss = tf.reduce_mean(tf.abs(y_true_edges - y_pred_edges))
    # Combine losses
    total_loss = bce + edge_loss
    return total_loss

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, gama=2):
        super(DiceLoss, self).__init__()
        self.name = 'NDL'
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result

def combined_bce_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = DiceLoss()
    return bce + dice(y_true, y_pred)

# Define learning rate schedule
decay_steps = (input_train.shape[0] // 32) * 10
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,  # Reduce initial learning rate
    decay_steps=decay_steps,
    decay_rate=0.96,
    staircase=True
)
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss = DiceLoss()

model.compile(optimizer=adam_optimizer, loss=loss, metrics=['accuracy', tf.keras.metrics.MeanAbsoluteError()])
history = model.fit(input_train, output_train, epochs=200, batch_size=16, validation_data=(input_val, output_val))
model.save(f"./plots_loss/hybrid_{test_n}_NEW")

# %%

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"plots_loss/loss_hybrid_{test_n}_NEW.png")  # Save the plot as an image
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f"plots_loss/accuracy_hybrid_{test_n}_NEW.png")  # Save the plot as an image

# %%

y = model.predict(input_val)

index = 50
plt.ion() 
fig,ax = plt.subplots(1,2)
ax[0].imshow(np.array(-y[index]).reshape(60, 60).T, cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[0].set_title('Predicted')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].matshow(-output_val[index].reshape(60, 60).T, cmap='gray')
ax[1].set_title('Expected')
ax[1].set_xticks([])
ax[1].set_yticks([])
fig.show()

# %%

# Save training and validation loss to a CSV file
loss_data = {
    'Training Loss': history.history['loss'],
    'Validation Loss': history.history['val_loss'],
    'Training Accuracy': history.history['accuracy'],
    'Validation Accuracy': history.history['val_accuracy']
}
loss_df = pd.DataFrame(loss_data)
loss_df.to_csv(f"./plots_loss/loss_data_{test_n}_NEW.csv", index=False)