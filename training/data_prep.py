import warnings
from pathlib import Path

from datasets import load_dataset, Audio

from utils import *

warnings.filterwarnings("ignore")

dataset = load_dataset("rfcx/frugalai")
dataset = dataset.cast_column("audio", Audio())

train_data = dataset['train']
test_data = dataset['test']

BASE_DIR = Path('data/')
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"

BASE_DIR.mkdir(exist_ok=True)
TRAIN_DIR.mkdir(exist_ok=True)
TEST_DIR.mkdir(exist_ok=True)

label_mapping = {
    0: "chainsaw",
    1: "environment"
}

save_audio_by_label(train_data, str(TRAIN_DIR), label_mapping)
save_audio_by_label(test_data, str(TEST_DIR), label_mapping)

# Remove corrupted files
corrupted_files = check_audio_files(str(BASE_DIR))
remove_corrupted_files(corrupted_files)

print("Data preparation and cleaning complete!")