# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

# Cargar la imagen en escala de grises
# Create dummy input data
bc = np.loadtxt('../simp/results_merge_2/bc.txt')
load = np.loadtxt('../simp/results_merge_2/load.txt')
output = np.loadtxt('../simp/results_merge_2/output.txt')

# Generate random input data
input_shape = (61, 61)  # Input size of 61x61
num_channels = 2  # Number of channels in each input array
batch_size = bc.shape[0]  # Number of samples in each batch

input_data = np.zeros((batch_size,) + input_shape + (num_channels,))
for i in range(batch_size):
    input_data[i, :, :, 0] = bc[i].reshape((61,61))
    input_data[i, :, :, 1] = load[i].reshape((61,61))

output_train = output.reshape(output.shape[0], 60, 60)
input_test = input_data[-1000:]
output_test = output_train[-1000:]
# %%

# Aplicar la transformada de Fourier bidimensional
img_fft = fft2(output_train)
img_fft_shifted = fftshift(img_fft)  # Centrar la frecuencia cero
print(img_fft_shifted.shape)
print(img_fft.shape)

# %%
# Visualizar la imagen original y su transformada de Fourier
index = 0
plt.subplot(121), plt.imshow(output_train[index], cmap='gray')
plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(np.log(1 + np.abs(img_fft_shifted[index])), cmap='gray')
plt.title('Transformada de Fourier'), plt.xticks([]), plt.yticks([])
plt.show()