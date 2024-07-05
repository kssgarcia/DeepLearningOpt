# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

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

input_train = input_data[-1000:]
output_train = output_data[-1000:]

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

class Patches(nn.Module):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def forward(self, images):
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(images.size(0), -1, self.patch_size * self.patch_size * images.size(1))
        return patches

class PatchEncoder(nn.Module):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        # Assuming each patch has been flattened to 1152 dimensions as per your input
        self.projection = nn.Linear(1152, projection_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches, projection_dim))

    def forward(self, patches):
        # patches should have shape [batch_size, num_patches, patch_size * patch_size * num_channels]
        # which is [16, 25, 1152] as per your input shape
        encoded = self.projection(patches) + self.position_embedding
        return encoded

class MLP(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(MLP, self).__init__()
        layers = []
        for units in hidden_units:
            layers.append(nn.Linear(units, units))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class TransformerBlock(nn.Module):
    def __init__(self, projection_dim, num_heads, transformer_units, transformer_layers):
        super(TransformerBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(transformer_layers):
            self.layers.append(nn.TransformerEncoderLayer(projection_dim, num_heads, dim_feedforward=transformer_units[0], dropout=0.1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EncodingBlockSkip(nn.Module):
    def __init__(self, filters_input, filters, stride):
        super(EncodingBlockSkip, self).__init__()
        self.conv1 = nn.Conv2d(filters_input, filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(filters)
        self.batchnorm2 = nn.BatchNorm2d(filters)
        self.dropout = nn.Dropout(0.1)
        self.pool = nn.MaxPool2d(kernel_size=stride) if stride > 1 else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x

class DecodingBlockSkip(nn.Module):
    def __init__(self, filters_input, filters, stride):
        super(DecodingBlockSkip, self).__init__()
        total_filters = filters_input + filters  # Adjust this if different
        self.conv2DTranspose = nn.ConvTranspose2d(total_filters, filters, kernel_size=stride, stride=stride, padding=0)
        self.conv2D = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(filters)
        self.batchnorm2 = nn.BatchNorm2d(filters)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, concat_layer):
        print(x.shape, concat_layer.shape)
        x = torch.cat([x, concat_layer], dim=1)
        x = self.conv2DTranspose(x)
        x = torch.relu(x)
        x = self.batchnorm1(x)
        x = self.conv2D(x)
        x = torch.relu(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        return x

class HybridModel(nn.Module):
    def __init__(self, patch_size, projection_dim, num_heads, transformer_units, transformer_layers):
        super(HybridModel, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.encoded1 = EncodingBlockSkip(32, 64, 2)
        self.encoded2 = EncodingBlockSkip(64, 128, 2)
        self.encoded3 = EncodingBlockSkip(128, 256, 3)
        self.patches = Patches(patch_size)
        self.patchencoder = PatchEncoder(25, projection_dim)
        self.transformerblock = TransformerBlock(projection_dim, num_heads, transformer_units, transformer_layers)
        self.decoded1 = DecodingBlockSkip(128, 256, 3)
        self.decoded2 = DecodingBlockSkip(256, 128, 2)
        self.decoded3 = DecodingBlockSkip(128, 64, 2)
        self.last = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Changed to 64 filters, kernel size to 3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Kept 64 filters, kernel size to 3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # Reduced to 32 filters, kernel size to 3
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # Reduced to 16 filters, kernel size to 3
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.output_tensor = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        initial = self.initial(x)
        print(initial.shape)
        encoded1 = self.encoded1(initial)
        print(encoded1.shape)
        encoded2 = self.encoded2(encoded1)
        print(encoded2.shape)
        encoded3 = self.encoded3(encoded2)
        print(encoded3.shape)
        
        x = self.patches(encoded2)
        print(x.shape)
        x = self.patchencoder(x)
        print(x.shape)
        x = self.transformerblock(x)
        print(x.shape)
        
        num_patches = int(np.sqrt(x.shape[1]))
        x = x.view(x.size(0), x.size(2), num_patches, num_patches)
        
        print(x.shape)
        x = self.decoded1(x, encoded3)
        print(x.shape)
        x = self.decoded2(x, encoded2)
        print(x.shape)
        x = self.decoded3(x, encoded1)
        print(x.shape)
        x = self.last(x)
        print(x.shape)
        x = self.output_tensor(x)
        print(x.shape)
        
        return x

test_n = 1

# Hyperparameters
patch_size = 3
projection_dim = 128
num_heads = 8
transformer_units = [
    projection_dim * 2,
    projection_dim
]
transformer_layers = 4

# Initialize model
model = HybridModel(patch_size, projection_dim, num_heads, transformer_units, transformer_layers).to(device)

# Optimizer and loss function
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 200
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
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
    
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/total_train:.4f}, Val Loss: {val_loss/total_val:.4f}, Train Acc: {train_correct/total_train:.4f}, Val Acc: {val_correct/total_val:.4f}')

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
plt.savefig(f"plots/loss_hybrid_{test_n}.png")  # Save the plot as an image
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

# Prediction and visualization
model.eval()
with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        y = outputs.cpu().numpy()
        break

index = 50
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

# Save training and validation loss to a CSV file
loss_data = {
    'Training Loss': train_losses,
    'Validation Loss': val_losses,
    'Training Accuracy': train_accuracies,
    'Validation Accuracy': val_accuracies
}
loss_df = pd.DataFrame(loss_data)
loss_df.to_csv(f"./plots_loss/loss_data_{test_n}.csv", index=False)
