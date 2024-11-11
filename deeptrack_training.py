import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# Define the NGSIM or synthetic data loader
class NGSIMDataset(Dataset):
    def _init_(self, data_path=None, num_samples=10000, seq_length=50):
        self.seq_length = seq_length

        if data_path:  # Use real or synthetic NGSIM data
            self.data = pd.read_csv(data_path)
            self.samples = self._process_ngsim_data()
        else:  # Generate synthetic data if no path is provided
            self.samples = self._generate_synthetic_data(num_samples)

        # Normalize the data
        self.samples = self._normalize_samples(self.samples)

    def _process_ngsim_data(self):
        samples = []
        vehicle_ids = self.data['vehicle_id'].unique()

        for vehicle_id in vehicle_ids:
            vehicle_data = self.data[self.data['vehicle_id'] == vehicle_id]
            for i in range(len(vehicle_data) - self.seq_length + 1):
                seq_data = vehicle_data.iloc[i:i + self.seq_length]
                nbr_traj = seq_data[['nbr_x', 'nbr_y']].values  # Neighbor trajectories
                ego_traj = seq_data[['ego_x', 'ego_y']].values  # Ego trajectory
                samples.append((nbr_traj, ego_traj))
        return samples

    def _generate_synthetic_data(self, num_samples):
        samples = []
        for _ in range(num_samples):
            nbr_traj = np.random.rand(self.seq_length, 2)  # Neighbor's trajectory (random x, y)
            ego_traj = np.random.rand(self.seq_length, 2)  # Ego vehicle trajectory (random x, y)
            samples.append((nbr_traj, ego_traj))
        return samples

    def _normalize_samples(self, samples):
        # Normalize samples to the range [0, 1]
        normalized_samples = []
        for nbr_traj, ego_traj in samples:
            nbr_traj = (nbr_traj - np.min(nbr_traj, axis=0)) / (np.max(nbr_traj, axis=0) - np.min(nbr_traj, axis=0))
            ego_traj = (ego_traj - np.min(ego_traj, axis=0)) / (np.max(ego_traj, axis=0) - np.min(ego_traj, axis=0))
            normalized_samples.append((nbr_traj, ego_traj))
        return normalized_samples

    def _len_(self):
        return len(self.samples)

    def _getitem_(self, idx):
        nbr_traj, ego_traj = self.samples[idx]
        nbr_traj = torch.tensor(nbr_traj, dtype=torch.float32)
        ego_traj = torch.tensor(ego_traj, dtype=torch.float32)
        return nbr_traj, ego_traj

# Define the TCN model
class TCN(nn.Module):
    def _init_(self, input_dim=2, output_dim=2, num_channels=[32, 64, 128], dropout_rate=0.2):
        super(TCN, self)._init_()
        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)

        # Creating TCN layers
        for out_channels in num_channels:
            self.convs.append(nn.Conv1d(input_dim, out_channels, kernel_size=3, padding=1))
            self.relus.append(nn.ReLU())
            self.batch_norms.append(nn.BatchNorm1d(out_channels))
            input_dim = out_channels

        self.fc = nn.Linear(num_channels[-1] * 2, output_dim)

    def forward(self, nbr_traj, ego_traj):
        nbr_traj = nbr_traj.permute(0, 2, 1)  # (batch_size, input_dim, seq_length)
        ego_traj = ego_traj.permute(0, 2, 1)

        for conv, relu, batch_norm in zip(self.convs, self.relus, self.batch_norms):
            nbr_traj = relu(batch_norm(conv(nbr_traj)))
            ego_traj = relu(batch_norm(conv(ego_traj)))
            nbr_traj = self.dropout(nbr_traj)
            ego_traj = self.dropout(ego_traj)

        nbr_traj = nbr_traj[:, :, -1]  # Take the last time step
        ego_traj = ego_traj[:, :, -1]

        combined_traj = torch.cat((nbr_traj, ego_traj), dim=1)  # Concatenate along the channel dimension
        output = self.fc(combined_traj)
        return output

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))

# Training function with test evaluation after each epoch
def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs, scheduler=None):
    model.train()
    train_losses = []
    test_mse_scores = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for nbr_traj, ego_traj in train_dataloader:
            optimizer.zero_grad()
            outputs = model(nbr_traj, ego_traj)
            loss = criterion(outputs, ego_traj[:, -1, :])  # Predicting the final position
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Compute average training loss
        avg_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_loss)

        # Evaluate on test set and calculate MSE
        test_mse = evaluate_model(model, test_dataloader)
        test_mse_scores.append(test_mse)

        # Adjust the learning rate based on the validation loss
        if scheduler:
            scheduler.step(test_mse)  # Pass the test MSE to the scheduler

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Test MSE: {test_mse:.4f}')

    return train_losses, test_mse_scores

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    distance_threshold = 12.0  # Set threshold to 12 units for accuracy check

    with torch.no_grad():
        for nbr_traj, ego_traj in dataloader:
            outputs = model(nbr_traj, ego_traj)
            total_loss += nn.functional.mse_loss(outputs, ego_traj[:, -1, :]).item()

            # Check how many predictions are correct
            for output, true_value in zip(outputs, ego_traj[:, -1, :]):
                if torch.norm(output - true_value) <= distance_threshold:
                    correct_predictions += 1
                    print(f"Prediction: {output.numpy()}, True Value: {true_value.numpy()} -> Correct")
                else:
                    print(f"Prediction: {output.numpy()}, True Value: {true_value.numpy()} -> Wrong")
                total_samples += 1

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    return total_loss / len(dataloader)

# Main code execution
if _name_ == '_main_':
    # Specify the path to synthetic or real NGSIM data
    data_path = "/content/drive/MyDrive/ngsim.csv"  # Replace with actual path if needed
    seq_length = 50
    dataset = NGSIMDataset(data_path=data_path, seq_length=seq_length)

    # Create dataset and dataloaders with 70:10:20 split for train, evaluate, and test
    train_size = int(0.7 * len(dataset))
    eval_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - eval_size
    train_dataset, eval_dataset, test_dataset = random_split(dataset, [train_size, eval_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, criterion, and optimizer
    input_dim = 2  # x and y coordinates
    output_dim = 2  # Predicting x and y coordinates
    num_channels = [32, 64, 128]  # Number of filters for each TCN layer
    model = TCN(input_dim, output_dim, num_channels)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    # Train the model and track both train loss and test MSE
    train_losses, test_mse_scores = train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=20, scheduler=scheduler)

    # Plot the training loss and test MSE over epochs
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_mse_scores, label='Test MSE')
    plt.title('Training Loss and Test MSE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / MSE')
    plt.legend()
    plt.grid()
    plt.show()
