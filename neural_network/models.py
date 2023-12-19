from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten
from tensorflow.keras import layers
import tensorflow as tf

# %% CNN architecture
def CNN_model(num_channels):
    model = Sequential()

    # Layers down-sampling
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(61, 61, num_channels)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((3, 3)))

    # Upsampling layers
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((3, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (1, 1), activation='sigmoid'))

    # Output layer
    return model

# %% CNN with U-NET architecture
def UNN_model(num_channels):
    # Input layer
    input_tensor = Input(shape=(61, 61, num_channels))
    print("After Initial Convolution:", input_tensor.shape)

    # Initial Convolution Layer (No Padding)
    initial = Conv2D(32, kernel_size=(2, 2), activation='relu', padding='valid')(input_tensor)
    initial = BatchNormalization()(initial)

    # Encoding Blocks
    def encoding_block(input_layer, filters):
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        if x.shape[1] == 15:
            encoded = MaxPooling2D((3, 3))(x)
        else:
            encoded = MaxPooling2D((2, 2))(x)

        encoded = layers.Dropout(rate=0.2)(encoded) # New line
        return encoded

    encoded1 = encoding_block(initial, filters=64)
    encoded2 = encoding_block(encoded1, filters=128)
    encoded3 = encoding_block(encoded2, filters=256)

    # Additional Convolution Layers for Feature Maps
    x = Conv2D(256, kernel_size=(7, 7), activation='relu', padding='same')(encoded3)
    x = BatchNormalization()(x)
    x = Conv2D(256, kernel_size=(7, 7), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # Decoding Blocks
    def decoding_block(input_layer, concat_layer, filters):
        x = concatenate([input_layer, concat_layer], axis=-1)
        if x.shape[1] == 5:
            x = Conv2DTranspose(filters, (3, 3), strides=(3,3), activation='relu', padding='same')(x)
        else:
            x = Conv2DTranspose(filters, (2, 2), strides=(2,2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        x = layers.Dropout(rate=0.2)(x) # New line
        return x

    decoded1 = decoding_block(x, encoded3, filters=128)
    decoded2 = decoding_block(decoded1, encoded2, filters=64)
    decoded3 = decoding_block(decoded2, encoded1, filters=32)

    # Final Convolution Layers for Element Solution
    x = concatenate([decoded3, initial], axis=-1)
    x = Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # Output layer with Sigmoid activation for binary classification
    output_tensor = Conv2D(1, (1, 1), activation='sigmoid')(x)
    print("After Final:", output_tensor.shape)

    # Create the model
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()
    return model

# %% ViT architecture
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

def ViT_model(input_shape, patch_size, num_patches, projection_dim, num_heads, transformer_units, transformer_layers):
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
