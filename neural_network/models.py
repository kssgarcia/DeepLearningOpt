# %%
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
import numpy as np

# %% CNN architecture
def CNN_model(input_shape):
    model = Sequential()

    # Layers down-sampling
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
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
def UNN_model(input_shape):
    # Input layer
    input_tensor = Input(shape=input_shape)
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
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def decoding_block(input_layer, filters, strides, trainable=False):
    if strides>1:
        x = layers.Conv2DTranspose(filters, (strides, strides), strides=(strides, strides), activation='relu', padding='same', trainable=trainable)(input_layer)
    else:
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', trainable=trainable)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', trainable=trainable)(x)
    x = layers.BatchNormalization()(x)
    return x

def ViT_model(input_shape, patch_size, num_patches, projection_dim, num_heads, transformer_units, transformer_layers):
    inputs = layers.Input(shape=input_shape)
    initial = layers.Conv2D(inputs.shape[-1], kernel_size=(2, 2), activation='relu', padding='valid')(inputs)
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

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    reshape_dim = int(np.sqrt(representation.shape[1]))
    resize1 = tf.reshape(representation, [-1, reshape_dim, reshape_dim, projection_dim])

    decoded1 = decoding_block(resize1, 64, 3, True)
    decoded2 = decoding_block(decoded1, 64, 1, True)
    decoded3 = decoding_block(decoded2, 32, 2, True)
    print(decoded3.shape)

    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(decoded3)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    output_tensor = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=output_tensor)
    model.summary()
    return model

# %%

class PatchEncoderPVT(layers.Layer):
    def __init__(self, image_size, patch_size, projection_dim):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def transformer_block(x, num_heads, projection_dim):
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ] 
    H = x.shape[1]
    patch_size = 3
    if H%3 != 0:
        patch_size = 2

    # Create patches.
    patches = Patches(patch_size)(x)
    # Encode patches.
    encoded_patches = PatchEncoderPVT(H, patch_size, projection_dim)(patches)
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
    connection = layers.Add()([x3, x2])
    x = tf.reshape(connection, [-1, int(H/patch_size), int(H/patch_size), projection_dim])

    return x

def PVT_model(input_shape, num_heads):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(2, kernel_size=(2, 2), activation='relu', padding='valid')(inputs)

    # Create multiple layers of the Transformer block
    encoded_transform1 = transformer_block(x, num_heads, 64)
    encoded_transform2 = transformer_block(encoded_transform1, num_heads, 128)
    encoded_transform3 = transformer_block(encoded_transform2, num_heads, 256)

    # Create multiple layers of the decoding block
    decoded1 = decoding_block(encoded_transform3, 128, 2, True)
    decoded2 = decoding_block(decoded1, 64, 2, True)
    decoded3 = decoding_block(decoded2, 32, 3, True)

    output_tensor = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoded3)

    model = keras.Model(inputs=inputs, outputs=output_tensor)

    model.summary()
    return model

# %% DETR

def DETR_model(input_shape, patch_size, num_patches, projection_dim, num_heads, transformer_units, transformer_layers):
    inputs = layers.Input(shape=input_shape)

    # Initial Convolution Layer (No Padding)
    initial = Conv2D(32, kernel_size=(2, 2), activation='relu', padding='valid')(inputs)
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
    def decoding_blockdetr(input_layer, concat_layer, filters):
        x = concatenate([input_layer, concat_layer], axis=-1)
        if x.shape[1] == 5:
            x = Conv2DTranspose(filters, (3, 3), strides=(3,3), activation='relu', padding='same')(x)
        else:
            x = Conv2DTranspose(filters, (2, 2), strides=(2,2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        return x

    decoded1 = decoding_blockdetr(x, encoded3, filters=128)
    decoded2 = decoding_blockdetr(decoded1, encoded2, filters=64)
    decoded3 = decoding_blockdetr(decoded2, encoded1, filters=32)

    # Final Convolution Layers for Element Solution
    x = concatenate([decoded3, initial], axis=-1)
    x = Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # Create patches.
    patches = Patches(patch_size)(x)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    resize1 = tf.reshape(representation, [-1, 6, 6, projection_dim])

    decoded1 = decoding_block(resize1, 64, 5)
    decoded2 = decoding_block(decoded1, 32, 2)

    output_tensor = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoded2)

    model = keras.Model(inputs=inputs, outputs=output_tensor)
    model.summary()
    return model

if __name__ == '__main__':
    input_shape = (61, 61, 4)
    patch_size = 16
    num_patches = (256 // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [projection_dim * 2, projection_dim]
    transformer_layers = 8

    model = DETR_model(input_shape, patch_size, num_patches, projection_dim, num_heads, transformer_units, transformer_layers)

# %% DETR simple

def DETR_model_simple(input_shape, patch_size, num_patches, projection_dim, num_heads, transformer_units, transformer_layers):
    inputs = layers.Input(shape=input_shape)

    # Initial Convolution Layer (No Padding)
    initial = Conv2D(32, kernel_size=(2, 2), activation='relu', padding='valid')(inputs)
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
    def decoding_blockdetr(input_layer, concat_layer, filters):
        x = concatenate([input_layer, concat_layer], axis=-1)
        if x.shape[1] == 5:
            x = Conv2DTranspose(filters, (3, 3), strides=(3,3), activation='relu', padding='same')(x)
        else:
            x = Conv2DTranspose(filters, (2, 2), strides=(2,2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        return x

    decoded1 = decoding_blockdetr(x, encoded3, filters=128)
    decoded2 = decoding_blockdetr(decoded1, encoded2, filters=64)
    decoded3 = decoding_blockdetr(decoded2, encoded1, filters=32)

    # Final Convolution Layers for Element Solution
    x = concatenate([decoded3, initial], axis=-1)
    x = Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    encoded_transform1 = transformer_block(x, num_heads, 64)
    encoded_transform2 = transformer_block(encoded_transform1, num_heads, 128)
    encoded_transform3 = transformer_block(encoded_transform2, num_heads, 256)

    # Create multiple layers of the decoding block
    decoded1 = decoding_block(encoded_transform3, 128, 2, True)
    decoded2 = decoding_block(decoded1, 64, 2, True)
    decoded3 = decoding_block(decoded2, 32, 3, True)

    output_tensor = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoded3)

    model = keras.Model(inputs=inputs, outputs=output_tensor)
    model.summary()
    return model

# %% Hybrid model

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
        self.conv2D = layers.Conv2D(self.filters, (3, 3), activation='relu', padding='same')
        self.batchnorm2 = layers.BatchNormalization()

    def call(self, input_layer, concat_layer):
        x = self.concat([input_layer, concat_layer])
        x = self.conv2DTranspose(x)
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
        self.conv2 = layers.Conv2D(self.filters, (3, 3), activation='relu', padding='same')
        self.batchnorm2 = layers.BatchNormalization()
        if self.stride > 1:
            self.pool = layers.MaxPooling2D((self.stride, self.stride))
        else:
            self.batchnorm3 = layers.BatchNormalization()

    def call(self, input_layer):
        x = self.conv1(input_layer)
        x = self.batchnorm1(x)
        x = self.conv2(x)
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
            layers.Conv2D(32, kernel_size=(2, 2), activation='relu', padding='valid'),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization()
        ])
        self.encoded1 = EncodingBlockSkip(64, 2)
        self.encoded2 = EncodingBlockSkip(128, 2)
        self.encoded3 = EncodingBlockSkip(256, 1)
        self.patches = Patches(self.patch_size)
        self.patchencoder = PatchEncoder(25, self.projection_dim)
        self.transformerblock = TransformerBlock(self.projection_dim, self.num_heads, self.transformer_units, self.transformer_layers)
        self.decoded1 = DecodingBlock(256, 3, True)
        self.decoded2 = DecodingBlockSkip(128, 2)
        self.decoded3 = DecodingBlockSkip(64, 2)
        self.decoded4 = DecodingBlockSkip(32, 1)
        self.last = keras.Sequential([
            layers.Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same'),
            layers.BatchNormalization()
        ])
        self.output_tensor = layers.Conv2D(1, (1, 1), activation='sigmoid')

    def call(self, inputs):
        initial = self.initial(inputs)
        encoded1 = self.encoded1(initial)
        encoded2 = self.encoded2(encoded1)
        encoded3 = self.encoded3(encoded2)
        
        self.num_patches = (encoded3.shape[2] // self.patch_size) ** 2

        x = self.patches(encoded3)
        x = self.patchencoder(x)

        x = self.transformerblock(x)

        reshape_dim = int(np.sqrt(x.shape[1]))
        x = tf.reshape(x, [-1, reshape_dim, reshape_dim, self.projection_dim])

        x = self.decoded1(x)
        x = self.decoded2(x, encoded2)
        x = self.decoded3(x, encoded1)
        x = self.decoded4(x, initial)

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