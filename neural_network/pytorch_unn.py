import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import logging
import time

# Setup logging
logging.basicConfig(filename='./plots_loss/training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
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
output_data = x.reshape((x.shape[0], 60, 60))

input_train = input_data[:1000]
output_train = output_data[:1000]

input_val = input_data[-100:]
output_val = output_data[-100:]

batch_size = input_train.shape[0]

# Normalize
output_val = np.where(output_val > 0.5, 1, 0)
output_train = np.where(output_train > 0.5, 1, 0)

if np.any((output_val > 0) & (output_val < 1)):
    print("output_val has elements between 0 and 1")
else:
    print("output_val does not have elements between 0 and 1")

output_train = output_train[:, np.newaxis, :, :]
output_val = output_val[:, np.newaxis, :, :]

class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        # Ensure input tensors are properly formatted for PyTorch: batch_size, channels, height, width
        self.inputs = torch.tensor(inputs, dtype=torch.float32).permute(0, 3, 1, 2)
        # No permutation needed for targets because the new axis for channel is already in the correct place
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Now, initialize your dataset with the reshaped data
train_dataset = CustomDataset(input_train, output_train)
val_dataset = CustomDataset(input_val, output_val)

# DataLoader remains unchanged
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class UNN_Model(nn.Module):
    def __init__(self):
        super(UNN_Model, self).__init__()

        # Initial Convolution
        self.initial_conv = nn.Conv2d(4, 32, kernel_size=(2, 2), padding=0)
        self.initial_bn = nn.BatchNorm2d(32)

        # Encoding Blocks
        self.enc_block1 = self.encoding_block(32, 64)
        self.enc_block2 = self.encoding_block(64, 128)
        self.enc_block3 = self.encoding_block(128, 256)

        # Additional Convolution Layers for Feature Maps
        self.additional_conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), padding=3)
        self.additional_bn1 = nn.BatchNorm2d(256)
        self.additional_conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), padding=3)
        self.additional_bn2 = nn.BatchNorm2d(256)

        # Decoding Blocks
        self.dec_block1 = self.decoding_block(256, 256, 128)
        self.dec_block2 = self.decoding_block(128, 128, 64)
        self.dec_block3 = self.decoding_block(64, 64, 32)

        # Final Convolution
        self.final_conv1 = nn.Conv2d(64, 32, kernel_size=(7, 7), padding=3)
        self.final_bn1 = nn.BatchNorm2d(32)
        self.final_conv2 = nn.Conv2d(32, 1, kernel_size=(1, 1))
    
    def encoding_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def decoding_block(self, in_channels, concat_channels, out_channels):
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels + concat_channels, out_channels, kernel_size=(3, 3), stride=2, padding=1, output_padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial Convolution
        initial = self.initial_bn(F.relu(self.initial_conv(x)))

        # Encoding Path
        encoded1 = F.max_pool2d(self.enc_block1(initial), 2)
        encoded2 = F.max_pool2d(self.enc_block2(encoded1), 2)
        encoded3 = F.max_pool2d(self.enc_block3(encoded2), 2)

        # Additional Convolution Layers for Feature Maps
        x = self.additional_bn1(F.relu(self.additional_conv1(encoded3)))
        x = self.additional_bn2(F.relu(self.additional_conv2(x)))

        # Decoding Path
        decoded1 = self.dec_block1(x)
        decoded2 = self.dec_block2(decoded1)
        decoded3 = self.dec_block3(decoded2)

        # Final Convolution
        x = self.final_bn1(F.relu(self.final_conv1(decoded3)))
        output_tensor = torch.sigmoid(self.final_conv2(x))

        return output_tensor

# Instantiate the model
model = UNN_Model()
print(model)

# %% Train Model

# Optimizer and loss function
criterion = nn.BCEWithLogitsLoss()
lr=1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
epochs = 100
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
ud = []

for epoch in range(epochs):
    t0 = time.time()
    model.train()
    train_loss = 0.0
    train_correct = 0
    total_train = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        train_correct += (predicted == targets).sum().item()
        total_train += targets.numel()
        
    train_losses.append(train_loss / total_train)
    train_accuracies.append(train_correct / total_train)
    
    model.eval()
    val_loss = 0.0
    val_correct = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            val_correct += (predicted == targets).sum().item()
            total_val += targets.numel()
            
    val_losses.append(val_loss / total_val)
    val_accuracies.append(val_correct / total_val)

    with torch.no_grad():
        ud.append([((lr * p.grad).std() / p.data.std()).log10().item() for p in model.parameters() if p.grad is not None])

    t1 = time.time()
    dt = (t1 - t0)
    logging.info(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/total_train:.4f} | Val Loss: {val_loss/total_val:.4f} | Train Acc: {train_correct/total_train:.4f} | Val Acc: {val_correct/total_val:.4f} | Time: {dt:.2f}ms')   
    print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/total_train:.4f} | Val Loss: {val_loss/total_val:.4f} | Train Acc: {train_correct/total_train:.4f} | Val Acc: {val_correct/total_val:.4f} | Time: {dt:.2f}ms')

# Save the model
torch.save(model.state_dict(), f"./plots_loss/hybrid_{test_n}.pt")

# Plotting training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"plots_loss/loss_hybrid_{test_n}.png")  # Save the plot as an image
plt.show()

# Plotting training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f"plots_loss/accuracy_hybrid_{test_n}.png")  # Save the plot as an image

# %% Eval Model

# Prediction and visualization
model.eval()
with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        y = outputs.cpu().numpy()
        break

index = 0
fig, ax = plt.subplots(1, 2)
ax[0].imshow(-y[index].reshape(60, 60).T, cmap='gray', interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
ax[0].set_title('Predicted')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(-output_val[index].reshape(60, 60).T, cmap='gray')
ax[1].set_title('Expected')
ax[1].set_xticks([])
ax[1].set_yticks([])
plt.show()
plt.savefig(f"plots_loss/hybrid_{test_n}.png")  # Save the plot as an image

# %% Plotting the output paramets

plt.figure(figsize=(20, 4))  # width and height of the plot
legends = []

# Iterate over each parameter and plot its gradient
for name, param in model.named_parameters():
    if name.split('.')[-1] == "weight":
        if param.requires_grad and param.grad is not None:
            t = param.cpu().detach()
            if len(t.shape) == 2:
                print(f'{name}: mean {t.mean():+f}, std {t.std():e}')
                hy, hx = torch.histogram(t, density=True)
                plt.plot(hx[:-1].detach(), hy.detach())
                legends.append(name)

plt.legend(legends)
plt.title('Weights Distribution')
plt.xlabel('Gradient value')
plt.ylabel('Density')
plt.show()

# %% Plotting the gradients

plt.figure(figsize=(20, 4))  # width and height of the plot
legends = []

# Iterate over each parameter and plot its gradient
for name, p in model.named_parameters():
    print(name)
    if name.split('.')[-1] == "weight":
        if p.requires_grad and p.grad is not None:
            t = p.grad.cpu().detach()
            if p.ndim == 2:
                print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
                hy, hx = torch.histogram(t, density=True)
                plt.plot(hx[:-1].detach(), hy.detach())
                legends.append(f'{i} {tuple(p.shape)}')

plt.legend(legends)
plt.title('Gradient Weights Distribution')
plt.xlabel('Gradient value')
plt.ylabel('Density')
plt.show()

# %% Plotting parameters update ratios

plt.figure(figsize=(20, 4))  # width and height of the plot
legends = []

plt.figure(figsize=(20, 4))
legends = []
for i,p in enumerate(range(len(ud))):
    plt.plot([ud[j][i] for j in range(len(ud)) if ud[j][i] is not None])
    legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
plt.legend(legends);