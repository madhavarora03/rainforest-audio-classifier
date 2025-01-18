import json
import warnings
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from tqdm.auto import tqdm

from utils import *

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Working on \"{device}\"")

class AudioDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 transform=None,
                 target_length=8000) -> None:
        """
        Initializes the AudioDataset.

        Args:
            root_dir (str): Root directory containing the dataset (e.g., 'data/train').
            transform (callable, optional): Transformation to apply to audio samples.
            target_length (int): Length of resampled samples
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.target_length = target_length
        self.class_to_idx = {
            name: idx for idx, name in enumerate(
                sorted(
                    [d.name for d in os.scandir(self.root_dir) if d.is_dir()]
                )
            )
        }

        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        self.audio_files = self._load_files()

    def _load_files(self) -> List[Tuple[str, int]]:
        """
        Loads all audio file paths and their corresponding labels.

        Returns:
            List of tuples containing (file_path, label_idx).
        """
        audio_files = []
        for class_name, label_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.root_dir, class_name)
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.endswith('.wav'):
                        file_path = os.path.join(root, file)
                        audio_files.append((file_path, label_idx))
        return audio_files

    def __len__(self) -> int:
        """
        Returns the total number of audio files in the dataset.

        Returns:
            int: Number of audio files.
        """
        return len(self.audio_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves the audio sample and label at the given index.

        Args:
            index (int): Index of the audio file.

        Returns:variables that may
            Tuple[torch.Tensor, int]: A tuple containing the audio waveform and its label.
        """
        file_path, label = self.audio_files[index]
        waveform, sample_rate = torchaudio.load(file_path)

        waveform = resize_audio(waveform, self.target_length)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label

BASE_DIR = Path('data/')
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"

train_dataset = AudioDataset(root_dir=str(TRAIN_DIR),
                             transform=None)

test_dataset = AudioDataset(root_dir=str(TEST_DIR),
                            transform=None)

print(f"Length of train dataset: {len(train_dataset)}")
print(f"Length of test dataset: {len(test_dataset)}")

first_waveform = train_dataset[0][0]
print(f"First waveform: \n{first_waveform}\nShape: {first_waveform.shape}")

BATCH_SIZE = 32
NUM_WORKERS = 0

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    shuffle=True
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    shuffle=False
)

waveforms, labels = next(iter(train_dataloader))

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


m5_model = M5(n_input=first_waveform.shape[0], n_output=len(train_dataset.class_to_idx.keys()))
m5_model.to(device)
print("\nModel Summary:")
summary(m5_model, input_size=waveforms.shape)
print()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        # Check if validation loss is nan
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: nn.Module = nn.CrossEntropyLoss(),
               device='cuda'):
    """
    Perform a single training step for the model.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): The DataLoader for training data.
        optimizer (Optimizer): The optimizer used to update the model parameters.
        loss_fn (nn.Module): The loss function, default is CrossEntropyLoss.
        device (str): The device to run the model on, 'cuda' or 'cpu'.

    Returns:
        train_loss (float): The total training loss for the step.
        train_acc (float): The average training accuracy for the step.
    """
    model.train()  # Set the model to training mode
    train_loss, train_acc = 0, 0
    correct, total = 0, 0

    for batch, (waveform, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc='Training'):
        X, y = waveform.to(device), label.to(device)

        # Forward pass
        y_pred = model(X).squeeze()

        # Compute loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Compute accuracy
        _, predicted = y_pred.max(1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate average loss and accuracy for the epoch
    avg_loss = train_loss / len(dataloader)
    avg_acc = correct / total

    return avg_loss, avg_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module = nn.CrossEntropyLoss(),
              device='cuda'):
    """
    Perform a single testing step for the model.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): The DataLoader for testing data.
        loss_fn (nn.Module): The loss function, default is CrossEntropyLoss.
        device (str): The device to run the model on, 'cuda' or 'cpu'.

    Returns:
        test_loss (float): The average test loss for the step.
        test_acc (float): The average test accuracy for the step.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss, correct, total = 0, 0, 0

    # Disable gradient calculations for testing
    with torch.inference_mode():
        for batch, (waveform, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc='Testing'):
            # Move data to the appropriate device
            X, y = waveform.to(device), label.to(device)

            # Forward pass
            y_pred = model(X).squeeze()

            # Compute loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # Compute accuracy
            _, predicted = y_pred.max(1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    # Calculate average loss and accuracy for the testing step
    avg_loss = test_loss / len(dataloader)
    avg_acc = correct / total

    return avg_loss, avg_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          loss_fn: nn.Module = nn.CrossEntropyLoss(),
          device: str = 'cuda',
          epochs: int = 10,
          patience: int = 5):
    """
    Train and evaluate the model over multiple epochs.

    Args:
        model (nn.Module): The neural network model.
        train_dataloader (DataLoader): The DataLoader for training data.
        test_dataloader (DataLoader): The DataLoader for testing/validation data.
        optimizer (Optimizer): The optimizer used to update the model parameters.
        scheduler (lr_scheduler._LRScheduler): The learning rate scheduler to adjust the learning rate.
        loss_fn (nn.Module): The loss function, default is CrossEntropyLoss.
        device (str): The device to run the model on, 'cuda' or 'cpu'.
        epochs (int): The number of epochs to train the model.
        patience (int): Number of epochs to wait before early stopping.

    Returns:
        dict: A dictionary containing training and testing losses and accuracies for each epoch.
    """
    # Define Early Stopping
    early_stopping = EarlyStopping(patience=patience, path=f"checkpoints/model_{model.__class__.__name__}.pth",
                                   verbose=True)

    # Store metrics for each epoch
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 30)

        # Perform a single training step
        train_loss, train_acc = train_step(model, train_dataloader, optimizer, loss_fn, device)

        # Perform a single testing step
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        # Save the metrics for this epoch
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        # Print out what's happenin'
        print(f"\nTrain Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f} | Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.6f}")

        # Apply learning rate scheduler step
        scheduler.step()

        # Apply Early Stopping
        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping...\n")
            break

    model.load_state_dict(torch.load(early_stopping.path))  # Load model with best weights available
    print("Training complete!")
    return history

baseline_model = M5(n_input=first_waveform.shape[0], n_output=len(train_dataset.class_to_idx.keys())).to(device)
optimizer = torch.optim.Adam(params=baseline_model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
loss_fn = nn.CrossEntropyLoss()

start = timer()
baseline_model_history = train(model=baseline_model,
                               train_dataloader=train_dataloader,
                               test_dataloader=test_dataloader,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               loss_fn=loss_fn,
                               device=device,
                               epochs=100)
end = timer()
baseline_model_train_time = print_train_time(start, end, device)

with open("model_training_history.json", "w") as f:
    json.dump(baseline_model_history, f)