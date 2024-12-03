import os
import torch
from torch.utils.data import Dataset, DataLoader
from music21 import midi, note, chord
import matplotlib.pyplot as plt


def open_midi(midi_path, remove_drums=True):
    """
    :param midi_path: path of MIDI file
    :param remove_drums: defaults to True. Removes drums track from MIDI file
    :return: returns MIDI stream
    """
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if remove_drums:
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]
    return midi.translate.midiFileToStream(mf)


def midi_to_binary_tensor(midi_path, remove_drums=True, time_resolution=0.25):
    """
    :param midi_path: path to MIDI file
    :param remove_drums: defaults to True. Removes drums track from MIDI file
    :param time_resolution: defaults 0.25. Represents what type of note the tensors will be divided into
    :return: returns torch.Tensor of what notes are being played per time_resolution
    """
    midi_stream = open_midi(midi_path, remove_drums)
    time_steps = int(midi_stream.flat.highestTime / time_resolution) + 1
    pitch_range = 128
    tensor = torch.zeros((time_steps, pitch_range), dtype=torch.int8)
    for part in midi_stream.parts:
        for note_or_chord in part.flat.notes:
            start_time = int(note_or_chord.offset / time_resolution)
            if isinstance(note_or_chord, note.Note):
                tensor[start_time, int(note_or_chord.pitch.ps)] = 1
            elif isinstance(note_or_chord, chord.Chord):
                for pitch in note_or_chord.pitches:
                    tensor[start_time, int(pitch.ps)] = 1
    return tensor

def plot_piano_roll(binary_tensor, title):
    """
    :param binary_tensor: binary torch tensor in CPU to be plotted
    """
    plt.imshow(binary_tensor.T, aspect='auto', cmap='gray')
    plt.xlabel('Time Steps')
    plt.ylabel('Pitch')
    plt.title(title)
    plt.show()


def process_midi_folder(midi_folder_path, remove_drums=True):
    """
    :param midi_folder_path: path to directory containing MIDI files
    :param remove_drums: defaults to True. Removes drums track from MIDI file
    :return: tensors of converted MIDI files truncated to the shape of the shortest MIDI file in directory, shortest length of a tensor in the directory
    """
    midi_files = [os.path.join(midi_folder_path, f) for f in os.listdir(midi_folder_path) if f.endswith('.mid')]

    # Find shortest length
    shortest_length = float('inf')
    midi_tensors = []

    for midi_path in midi_files:
        midi_tensor = midi_to_binary_tensor(midi_path, remove_drums=remove_drums)
        midi_tensors.append(midi_tensor)
        if midi_tensor.shape[0] < shortest_length:
            shortest_length = midi_tensor.shape[0]

    print(f"Shortest length among MIDI files: {shortest_length}")

    # Truncate all tensors to shortest_length
    truncated_tensors = []
    for tensor in midi_tensors:
        truncated_tensor = tensor[:shortest_length, :]  # Truncate to shortest length
        truncated_tensors.append(truncated_tensor)

    return truncated_tensors, shortest_length


# MidiTensorDataset for DataLoader
class MidiTensorDataset(Dataset):
    def __init__(self, tensor_list):
        self.tensor_list = tensor_list

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx):
        return self.tensor_list[idx]


def create_dataloader(truncated_tensors, batch_size=16, shuffle=True):
    """
    :param truncated_tensors: list of tensors of shape (time_steps, pitch_range)
    :param batch_size: number of samples (songs) per batch
    :param shuffle: Boolean to shuffle the samples
    :return: PyTorch DataLoader for the dataset
    """
    dataset = MidiTensorDataset(truncated_tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader