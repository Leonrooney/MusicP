# MusicP

A menu-driven ABC notation synthesizer with waveform selection, effects, and audio processing capabilities.

## Features

1. **Waveform Selection**: Choose from sine, square, sawtooth, or triangle waveforms
2. **Loudness Control**: Adjust master gain (0.0–1.0)
3. **ABC File Support**: Load and play ABC notation files
4. **Speed Control**: Adjust tempo (BPM)
5. **Pitch Shifting**: Shift pitch in semitones
6. **Background Noise**: Add white, pink, or brown noise with optional envelope
7. **External WAV Mixing**: Mix external audio files
8. **Playback**: Real-time audio playback
9. **Export**: Save synthesized audio as WAV files

## ABC Support

- **Headers**: X (reference number), T (title), K (key), C (composer), M (meter), Q (tempo), L (default note length)
- **Body**: Full note support (A–G/a–g), octave modifiers (commas/apostrophes), accidentals (^, =, _), rests (z), bar lines (|, ||), repeat bars (|:, :|)

## Requirements

- Python 3.10+
- numpy
- simpleaudio (optional, for playback)

## Installation

```bash
pip install numpy simpleaudio
```

## Usage

```bash
python3 abc_synth.py
```

Follow the menu prompts to configure your synthesizer and play ABC files.

## Example

The repository includes `test.abc` as a sample ABC notation file.

