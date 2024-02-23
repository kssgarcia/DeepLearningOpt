# %%
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from models import ViT_model

from resnet_backbone import ResNet50Backbone

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

def ViT_model1(input_shape, patch_size, num_patches, projection_dim, num_heads, transformer_units, transformer_layers):
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

if __name__=="__main__":
    learning_rate = 0.001
    weight_decay = 0.0001
    image_size = 60  # We'll resize input images to this size
    patch_size = 10  # Size of the patches to be extract from the input images
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 15
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 20
    input_shape = (61, 61, 3)

    model = ViT_model1(input_shape, patch_size, num_patches, projection_dim, num_heads, transformer_units, transformer_layers)
