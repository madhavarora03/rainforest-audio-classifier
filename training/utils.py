import os
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torchaudio
from colorama import Fore, Style, init
from datasets import IterableDataset
from torchaudio import transforms

# Initialize colorama
init(autoreset=True)


def save_audio_by_label(dataset: IterableDataset,
                        save_path: str,
                        label_to_dir: Dict[int, str]) -> None:
    """
    Save audio files from a dataset into separate directories based on their labels.

    Args:
        dataset (iterable): Streaming dataset containing audio files and labels.
        save_path (str): Root path where audio files will be saved.
        label_to_dir (dict): Mapping of label values to subdirectory names.

    Returns:
        None
    """
    print(f"\nSaving audio files to {save_path}...")
    for i, sample in enumerate(dataset):
        try:
            label = sample["label"]
            audio = sample["audio"]

            sub_dir = os.path.join(save_path, label_to_dir[label])
            os.makedirs(sub_dir, exist_ok=True)

            audio_path = os.path.basename(audio["path"])
            file_path = os.path.join(sub_dir, audio_path)

            sf.write(file_path, audio["array"], audio["sampling_rate"])

            if i % 1000 == 0 or i == len(dataset) - 1:
                print(f"{Fore.GREEN}Saved file #{i + 1}:{Style.RESET_ALL} {file_path}")

        except Exception as e:
            print(f"{Fore.RED}Error processing sample #{i + 1}:{Style.RESET_ALL} {sample}. Error: {e}")

    print(f"\nAudio saving completed.\n{'-'*25}")


def check_audio_files(directory: str) -> List[str]:
    """
    Recursively checks all .wav files in a directory for corruption.

    Args:
        directory (str): Path to the root directory to check.

    Returns:
        list: List of corrupted files (if any).
    """
    corrupted_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    waveform, sample_rate = torchaudio.load(file_path)
                except Exception as e:
                    print(
                        f"{Fore.RED}Corrupted file found:{Style.RESET_ALL} {file_path}. {Fore.RED}Error:{Style.RESET_ALL} {e}")
                    corrupted_files.append(file_path)

    if not corrupted_files:
        print("No corrupted files found.")
    else:
        print(f"\nFound {Fore.BLUE}{len(corrupted_files)}{Style.RESET_ALL} corrupted files.\n")

    return corrupted_files

def remove_corrupted_files(corrupted_files: List[str]) -> None:
    for file_path in corrupted_files:
        try:
            print(f"{Fore.MAGENTA}Handling corrupted file:{Style.RESET_ALL} {file_path}")
            os.remove(file_path)
            print(f"{Fore.RED}Deleted:{Style.RESET_ALL} {file_path}")
        except Exception as e:
            print(f"{Fore.RED}Failed to handle{Style.RESET_ALL} {file_path}. {Fore.RED}Error:{Style.RESET_ALL} {e}")


def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c + 1}")
        axes[c].set_xlabel("Time (seconds)")
    figure.suptitle("waveform")


def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c + 1}")
        axes[c].set_xlabel("Time (seconds)")
    figure.suptitle(title)


def print_train_time(start, end, device="cuda"):
    """Prints and returns the total training time."""
    total_time = end - start
    print(f"Training took {total_time:.3f} seconds on {device}.")
    return total_time


def visualize_random_waveform(dataset: torch.utils.data.Dataset, seed: int = None):
    """
    Visualizes a random waveform and its spectrogram from a dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to sample from. Each item should return
                                            (waveform, sample_rate, label, file_path).
        seed (int, optional): Random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

    # Select a random index and fetch the sample
    random_index = random.randint(0, len(dataset) - 1)
    waveform, sample_rate, label, file_path = dataset[random_index]

    # Convert waveform to numpy for plotting
    waveform_np = waveform.numpy()
    num_channels, num_frames = waveform_np.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    # Create subplots
    fig, axes = plt.subplots(num_channels, 2, figsize=(12, 6))
    if num_channels == 1:
        axes = [axes]

    # Plot waveform and spectrogram for each channel
    for c in range(num_channels):
        # Plot waveform
        axes[c][0].plot(time_axis, waveform_np[c], linewidth=1)
        axes[c][0].grid(True)
        axes[c][0].set_title(f"Waveform - Channel {c + 1}")
        axes[c][0].set_xlabel("Time (seconds)")
        axes[c][0].set_ylabel("Amplitude")

        # Plot spectrogram
        axes[c][1].specgram(waveform_np[c], Fs=sample_rate, scale='dB')
        axes[c][1].set_title(f"Spectrogram - Channel {c + 1}")
        axes[c][1].set_xlabel("Time (seconds)")
        axes[c][1].set_ylabel("Frequency (Hz)")

    fig.suptitle(f"Label: {dataset.idx_to_class[label]}\nFile: {file_path}", y=1.02, size=16)
    plt.tight_layout()
    plt.show()


def resize_audio(waveform, target_length):
    """Resizes the audio waveform to the target length using resampling"""
    num_frames = waveform.shape[-1]
    if num_frames != target_length:
        resampler = transforms.Resample(orig_freq=num_frames, new_freq=target_length)
        waveform = resampler(waveform)
    return waveform


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary."""
    # Get the loss values of the results dictionary(training and test)
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]

    # Get the accuracy values of the results dictionary (training and test)
    train_accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    # Figure out how mnay epochs there were
    epochs = range(len(results["train_loss"]))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    """
    Collate function to process a batch of audio samples.

    Args:
        batch (list): A list of tuples containing (waveform, sample_rate, label, file_path).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Batched waveforms and corresponding labels.
    """
    # Separate out the components of each data sample
    tensors, targets = [], []

    for waveform, label in batch:
        tensors.append(waveform)
        targets.append(label)

    # Pad waveforms to make them the same length
    tensors = pad_sequence(tensors)
    targets = torch.tensor(targets, dtype=torch.long)

    return tensors, targets