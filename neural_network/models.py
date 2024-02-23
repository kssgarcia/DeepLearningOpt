from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
import tensorflow as tf

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

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def decoding_block(input_layer, filters, strides, trainable=False):
    x = layers.Conv2DTranspose(filters, (strides, strides), strides=(strides, strides), activation='relu', padding='same', trainable=trainable)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', trainable=trainable)(x)
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

    decoded1 = decoding_block(resize1, 64, 5, True)
    decoded2 = decoding_block(decoded1, 32, 2, True)

    output_tensor = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoded2)

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

    backbone = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    #backbone = ResNet50Backbone(name='backbone')

    for layer in backbone.layers:
        layer.trainable = False

    backbone_block = backbone(inputs, training=False)

    decoded1 = decoding_block(backbone_block, 1792, 5)
    decoded2 = decoding_block(decoded1, 1536, 3)
    decoded3 = decoding_block(decoded2, 1280, 2)

    # Create patches.
    patches = Patches(patch_size)(decoded3)
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

    decoded1 = decoding_block(resize1, 64, 5)
    decoded2 = decoding_block(decoded1, 32, 2)

    output_tensor = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoded2)

    model = keras.Model(inputs=inputs, outputs=output_tensor)
    model.summary()
    return model

# %% DETR simple

def DETR_model_simple(input_shape, patch_size, num_patches, projection_dim, num_heads, transformer_units, transformer_layers):
    inputs = layers.Input(shape=input_shape)
    initial = layers.Conv2D(2, kernel_size=(2, 2), activation='relu', padding='valid')(inputs)

    down1 = MaxPooling2D((2, 2))(initial)
    down2 = Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape)(down1)
    down3 = MaxPooling2D((2, 2))(down2)
    down4 = Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape)(down3)
    down5 = MaxPooling2D((3, 3))(down4)

    up1 = UpSampling2D((3, 3))(down5)
    up2 = Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape)(up1)
    up3 = UpSampling2D((2, 2))(up2)
    up4 = Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape)(up3)
    up5 = UpSampling2D((2, 2))(up4)
    up6 = Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape)(up5)

    # Create patches.
    patches = Patches(patch_size)(up6)
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

    decoded1 = decoding_block(resize1, 64, 5)
    decoded2 = decoding_block(decoded1, 32, 2)

    output_tensor = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoded2)

    model = keras.Model(inputs=inputs, outputs=output_tensor)
    model.summary()
    return model
