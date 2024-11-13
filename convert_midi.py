import os
import torch
from music21 import midi, note, chord
from torch.utils.data import Dataset, DataLoader


def open_midi(midi_path, remove_drums):
    # There is an one-line method to read MIDIs
    # but to remove the drums we need to manipulate some
    # low level MIDI events.
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if (remove_drums):
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]

    return midi.translate.midiFileToStream(mf)


def midi_to_binary_tensor(midi_path, remove_drums=True, time_resolution=0.25):
    """
    Converts a MIDI file to a binary PyTorch tensor representation.

    Args:
    - midi_path: Path to the MIDI file.
    - remove_drums: Boolean indicating whether to remove drum tracks.
    - time_resolution: Time resolution in quarter notes (default 0.25 represents sixteenth notes).

    Returns:
    - A binary PyTorch tensor where:
      * Rows represent time (in quarter notes at given resolution).
      * Columns represent pitches (0-127).
      * Values are binary, 1 for note presence and 0 for absence.
    """
    midi_stream = open_midi(midi_path, remove_drums)
    time_steps = int(midi_stream.flat.highestTime / time_resolution) + 1
    pitch_range = 128  # MIDI pitch range (0-127)

    # Binary tensor to hold note on/off data
    tensor = torch.zeros((time_steps, pitch_range), dtype=torch.int8)

    for part in midi_stream.parts:
        for note_or_chord in part.flat.notes:
            start_time = int(note_or_chord.offset / time_resolution)

            if isinstance(note_or_chord, note.Note):
                pitch_value = int(note_or_chord.pitch.ps)
                tensor[start_time, pitch_value] = 1
            elif isinstance(note_or_chord, chord.Chord):
                for pitch in note_or_chord.pitches:
                    pitch_value = int(pitch.ps)
                    tensor[start_time, pitch_value] = 1

    return tensor


def midi_folder_to_tensors(midi_folder_path):
    # List down the midi files of the folder
    music_list = os.listdir(midi_folder_path)
    music_list = [f"{midi_folder_path}/{p}" for p in music_list]

    # Convert sequences to tensors
    tensors = list(map(midi_to_binary_tensor, music_list))

    return tensors


class MidiDataset(Dataset):
    def __init__(self, midi_folder_path, remove_drums=True, time_resolution=0.25):
        self.midi_folder_path = midi_folder_path
        self.midi_files = [os.path.join(midi_folder_path, f) for f in os.listdir(midi_folder_path) if f.endswith('.mid')]
        self.remove_drums = remove_drums
        self.time_resolution = time_resolution

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx):
        midi_path = self.midi_files[idx]
        tensor = midi_to_binary_tensor(midi_path, self.remove_drums, self.time_resolution)
        return tensor


def custom_collate_fn(batch):
    # Pads sequences in batch to the maximum sequence length in the batch
    max_time_steps = max(tensor.shape[0] for tensor in batch)
    pitch_range = 128  # Standard MIDI pitch range

    # Padding the tensors to have the same time dimension
    padded_batch = torch.zeros((len(batch), max_time_steps, pitch_range), dtype=torch.int8)

    for i, tensor in enumerate(batch):
        time_steps = tensor.shape[0]
        padded_batch[i, :time_steps, :] = tensor

    return padded_batch

def get_midi_data_loader(midi_folder_path, batch_size=4, shuffle=True, num_workers=0, remove_drums=True, time_resolution=0.25):
    dataset = MidiDataset(midi_folder_path, remove_drums, time_resolution)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=custom_collate_fn)
    return data_loader