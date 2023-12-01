# %%
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create dummy input data
bc = np.loadtxt('../simp/results_merge_2/bc.txt')
load = np.loadtxt('../simp/results_merge_2/load.txt')
#vol = np.loadtxt('../simp/results_merge_2/vol.txt')
output = np.loadtxt('../simp/results_merge_2/output.txt')

# Generate random input data
input_shape = (61, 61)  # Input size of 61x61
num_channels = 2  # Number of channels in each input array
batch_size = bc.shape[0]  # Number of samples in each batch

input_data = np.zeros((batch_size,) + input_shape + (num_channels,))
for i in range(batch_size):
    input_data[i, :, :, 0] = bc[i].reshape((61,61))
    input_data[i, :, :, 1] = load[i].reshape((61,61))
    #input_data[i, :, :, 2] = load[i].reshape((61,61))

output_train = output.reshape(output.shape[0], 60, 60, 1)

x_train = input_data[:-1000]
y_train = output_train[:-1000]
x_test = input_data[-1000:]
y_test = output_train[-1000:]

batch_size = x_train.shape[0]

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

# %%

num_classes = 100
input_shape = (61, 61, 2)

learning_rate = 0.001
weight_decay = 0.0001
num_epochs = 100
image_size = 60  # We'll resize input images to this size
patch_size = 10  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 12
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 15


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        H = patches.shape[1]
        patches = tf.reshape(patches, [batch_size, H*H, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def decoding_block(input_layer, filters):
    if input_layer.shape[1] == 6:
        x = layers.Conv2DTranspose(filters, (5, 5), strides=(5,5), activation='relu', padding='same')(input_layer)
    else:
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x

def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    initial = layers.Conv2D(2, kernel_size=(2, 2), activation='relu', padding='valid')(inputs)
    # Create patches.
    patches = Patches(patch_size)(initial)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    resize1 = tf.reshape(representation, [-1, 6, 6, projection_dim])

    decoded1 = decoding_block(resize1, filters=64)
    decoded2 = decoding_block(decoded1, filters=32)

    output_tensor = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoded2)

    model = keras.Model(inputs=inputs, outputs=output_tensor)
    model.summary()
    return model

model = create_vit_classifier()

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    './best/cp.ckpt',
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose= 1,
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_test, y_test, epochs=5, batch_size=10,validation_split=0.1, callbacks=[checkpoint_callback])
