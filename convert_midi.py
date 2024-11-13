import os
import torch
from music21 import midi, note, chord

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

