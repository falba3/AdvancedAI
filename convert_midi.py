import os
import torch
from music21 import midi, note, chord
from torch.utils.data import Dataset, DataLoader


def open_midi(midi_path, remove_drums):
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if remove_drums:
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]
    return midi.translate.midiFileToStream(mf)


def midi_to_binary_tensor(midi_path, remove_drums=True, time_resolution=0.25):
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


def tensor_to_chunks(tensor, chunk_size):
    time_steps, pitch_range = tensor.shape
    inputs, targets = [], []
    for start in range(0, time_steps - chunk_size):
        input_chunk = tensor[start:start + chunk_size]
        target_note = tensor[start + chunk_size]
        inputs.append(input_chunk)
        targets.append(target_note)
    return inputs, targets


class MidiDataset(Dataset):
    def __init__(self, midi_folder_path, chunk_size=32, remove_drums=True, time_resolution=0.25):
        self.midi_folder_path = midi_folder_path
        self.midi_files = [os.path.join(midi_folder_path, f) for f in os.listdir(midi_folder_path) if
                           f.endswith('.mid')]
        self.chunk_size = chunk_size
        self.remove_drums = remove_drums
        self.time_resolution = time_resolution

        # Map folder names to genre labels
        self.genre_map = {'rock_music': 0, 'pop_music': 1, 'classical_music': 2}

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx):
        midi_path = self.midi_files[idx]
        tensor = midi_to_binary_tensor(midi_path, self.remove_drums, self.time_resolution)
        inputs, targets = tensor_to_chunks(tensor, self.chunk_size)

        # Get the genre from the folder name
        genre_label = self.genre_map[os.path.basename(os.path.dirname(midi_path))]

        return inputs, targets, genre_label


def custom_collate_fn(batch):
    # Flatten the batch to have a list of (input, target) pairs across all songs
    inputs, targets, genre_labels = [], [], []
    for song_inputs, song_targets, genre_label in batch:
        inputs.extend(song_inputs)
        targets.extend(song_targets)
        genre_labels.extend([genre_label] * len(song_inputs))

    # Determine max time steps to pad each chunk in this batch
    max_time_steps = max(chunk.shape[0] for chunk in inputs)
    pitch_range = 128

    # Pad each chunk to the max time steps
    padded_inputs = torch.zeros((len(inputs), max_time_steps, pitch_range), dtype=torch.int8)
    for i, chunk in enumerate(inputs):
        padded_inputs[i, :chunk.shape[0], :] = chunk

    targets = torch.stack(targets)  # Stack all target notes
    genre_labels = torch.tensor(genre_labels)  # Genre labels for each chunk

    return padded_inputs, targets, genre_labels


def get_midi_data_loader(midi_folder_path, batch_size=4, chunk_size=32, shuffle=True, num_workers=0, remove_drums=True,
                         time_resolution=0.25):
    dataset = MidiDataset(midi_folder_path, chunk_size, remove_drums, time_resolution)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                             collate_fn=custom_collate_fn)
    return data_loader
