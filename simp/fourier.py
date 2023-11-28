# %%
from os import path, makedirs
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift

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
#inputbc_ff = fftshift(fft2(bc))
#inputload_ff = fftshift(fft2(load))
output_ff = fftshift(fft2(output_train))
print(output_ff)

#inputbc_rec = ifft2(ifftshift(output_ff))
#inputload_rec = ifft2(ifftshift(output_ff))
#output_rec = ifft2(ifftshift(output_ff))
# %%
# Save data
dir = './results_merge_2'
if not path.exists(dir): makedirs(dir)
np.savetxt(dir + '/fourier.txt', np.array(output_ff.flatten()), fmt='%.3f')

# %%
index = 10
plt.subplot(131), plt.imshow(-bc[index].reshape((61,61)), cmap='gray')
plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(np.log(1 + np.abs(inputbc_ff[index].reshape((61,61)))), cmap='gray')
plt.title('Transformada de Fourier'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(-np.abs(inputbc_rec[index]), cmap='gray')
plt.title('Reconstructed'), plt.xticks([]), plt.yticks([])
plt.show()

# %%
index = 10
plt.subplot(131), plt.imshow(-load[index], cmap='gray')
plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(np.log(1 + np.abs(inputload_ff[index])), cmap='gray')
plt.title('Transformada de Fourier'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(-np.abs(inputload_rec[index]), cmap='gray')
plt.title('Reconstructed'), plt.xticks([]), plt.yticks([])
plt.show()

# %%

# Visualizar la imagen original y su transformada de Fourier
index = 10
plt.subplot(131), plt.imshow(-output_train[index], cmap='gray')
plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(np.log(1 + np.abs(output_ff[index])), cmap='gray')
plt.title('Transformada de Fourier'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(-np.abs(output_rec[index]), cmap='gray')
plt.title('Reconstructed'), plt.xticks([]), plt.yticks([])
plt.show()