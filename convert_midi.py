import os
import numpy as np
import torch
import mido
import pretty_midi
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# Midi to Piano Roll Function
def midi_to_piano_roll(file_path, fs=100):
    midi_data = pretty_midi.PrettyMIDI(file_path)

    # Convert to piano roll (time x notes)
    piano_roll = midi_data.get_piano_roll(fs=fs)

    # Convert to binary (1 if note is played, 0 if not)
    piano_roll_binary = (piano_roll > 0).astype(int)
    return piano_roll_binary


# Piano Roll to Sequence Function
def roll2sequence(piano_roll, sequence_length=100):
    # Check if piano roll has enough time steps for at least one sequence
    if piano_roll.shape[1] < sequence_length:
        return None
    piano_roll_sequences = [piano_roll[:, i:i + sequence_length]
                        for i in range(0, piano_roll.shape[1] - sequence_length, sequence_length)]
    piano_roll_sequences = np.stack(piano_roll_sequences)
    return piano_roll_sequences # shape (num_sequences, 128, sequence_length)

# Sequence to Tensor Function
def roll2tensor(PR_sequences):
    return torch.tensor(PR_sequences, dtype=torch.float32) # shape (num_sequences, 128, sequence, length)


# Music Dataset class
class MusicDataset(Dataset):
    def __init__(self, tensor_list):
        self.data = tensor_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Padding function
def pad_collate(batch):
    # Pads each sequence to the length of the longest sequence in this batch
    batch = [item.squeeze() for item in batch]  # Remove extra dimensions if needed
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch

def midi_folder_2_loader(midi_folder, batch_size=32):
    # List down the midi files of the folder
    music_list = os.listdir(midi_folder)
    music_list = [f"{midi_folder}/{p}" for p in music_list]

    # Convert the midi files into piano rolls
    PR_list = list(map(midi_to_piano_roll, music_list))

    # Convert the piano rolls into sequences
    sequences = list(filter(None, map(roll2sequence, PR_list)))

    # Convert sequences to tensors
    tensors = list(map(roll2tensor, sequences))

    # Convert tensors to dataset
    dataset = MusicDataset(tensors)

    # Dataset to data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    return {midi_folder: loader}


def midi_folder_2_tensors(midi_folder):
    # List down the midi files of the folder
    music_list = os.listdir(midi_folder)
    music_list = [f"{midi_folder}/{p}" for p in music_list]

    # Convert the midi files into piano rolls
    PR_list = list(map(midi_to_piano_roll, music_list))

    # Convert the piano rolls into sequences
    # sequences = list(filter(None, map(roll2sequence, PR_list)))
    sequences = list( map(roll2sequence, PR_list))

    # Convert sequences to tensors
    tensors = list(map(roll2tensor, sequences))

    return {midi_folder: tensors}
