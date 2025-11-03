# ABC Music Synthesizer

**Author(s):** Leon Rooney  
**Student ID(s):** [Student ID]

## Overview of Program

This program is a menu-driven ABC notation synthesizer that converts ABC music notation files into synthesized audio and MIDI files. The synthesizer provides a comprehensive interface for audio manipulation, including waveform selection, volume control, tempo adjustment, pitch shifting, noise generation, and external audio mixing capabilities.

The program reads ABC notation files, parses the musical content (notes, rests, timing, key signatures, tempo), synthesizes the audio using digital signal processing techniques, and allows export to both WAV and MIDI formats.

## ABC Parsing and Assumptions

The ABC parser handles standard ABC notation with the following features:

### Headers Supported
- **X:** Reference number
- **T:** Title
- **K:** Key signature (e.g., "C", "G", "F#m")
- **C:** Composer
- **M:** Meter (e.g., "4/4", "3/4")
- **Q:** Tempo in BPM
- **L:** Default note length (e.g., "1/8", "1/4")

### Body Notation
- **Notes:** A–G (uppercase = one octave below middle C), a–g (lowercase = middle C octave)
- **Octave modifiers:** `'` (apostrophes) raise octave, `,` (commas) lower octave
- **Accidentals:** `^` (sharp), `_` (flat), `=` (natural)
- **Durations:** Numbers and fractions (e.g., "2" = double length, "/2" = half length, "3/2" = dotted quarter)
- **Rests:** `z` represents a rest
- **Bar lines:** `|` (single), `||` (double bar)
- **Repeat bars:** `|:` (start repeat), `:|` (end repeat) - with automatic playback of repeated sections

### Assumptions
- Key signatures default to major if not specified; minor keys are mapped to their relative major
- Default note length is 1/8 if not specified in L header
- Accidentals apply within a measure (reset at bar lines)
- Repeat bars automatically duplicate the enclosed section
- Pitch mapping: lowercase 'c' = MIDI 60 (middle C)

## Audio Synthesis (Waveforms, ADSR, BPM)

### Waveform Generation
Four waveforms are available:
- **Sine:** Smooth, fundamental frequency
- **Square:** Rich harmonics, digital sound
- **Sawtooth:** Bright, full harmonic spectrum
- **Triangle:** Softer than sawtooth, fewer harmonics

Waveforms are generated using phase accumulation techniques, where phase values (0–1) are computed from frequency and time, then mapped to the selected waveform function.

### ADSR Envelope
Every synthesized note applies an ADSR (Attack, Decay, Sustain, Release) envelope:
- **Attack:** 0.01 seconds - initial rise from silence
- **Decay:** 0.05 seconds - drop to sustain level
- **Sustain:** 0.8 (80%) - held level during note duration
- **Release:** 0.08 seconds - fade out after note ends

This creates natural-sounding note attacks and releases, avoiding clicks and abrupt transitions.

### BPM (Tempo) Control
The synthesizer supports tempo control in two ways:
1. From ABC file header (Q: directive)
2. Manual setting via menu option 4

Tempo directly affects note durations: `duration_seconds = (beats * 60) / BPM`. The menu BPM setting takes precedence over the file's Q value.

## Noise and Mixing

### Background Noise
Three noise types are available:
- **White noise:** Flat frequency spectrum, random at all frequencies
- **Pink noise:** 1/f noise, equal energy per octave (more natural)
- **Brown noise:** 1/f² noise, filtered white noise (deeper, rumbling)

Noise level is set as a ratio (0.0–1.0) relative to the signal peak. An optional triangular envelope can be applied to noise, creating a fade-in/fade-out effect over the entire audio duration.

### External WAV Mixing
The synthesizer can mix the ABC-generated audio with an external WAV file. The mix level (0.0–1.0) controls the balance between the ABC audio and the external file. The external file is automatically resampled to match the synthesis sample rate (44100 Hz) and trimmed/padded to match the ABC audio duration.

## MIDI Export (if implemented)

The MIDI export feature converts the parsed ABC score into a Standard MIDI File (.mid). Key implementation details:

- **Resolution:** 480 ticks per beat (standard MIDI resolution)
- **Tempo:** Set via meta message based on BPM setting
- **Velocity:** Mapped from loudness (0–1) to MIDI velocity range (20–110) for musical dynamics
- **Pitch shift:** Applied to note numbers before writing to MIDI
- **Note timing:** Durations converted from beats to ticks (beats × ticks_per_beat)
- **Rests:** Handled as silent time, accumulated and added to the delay of the next note-on event

The MIDI export requires the `mido` library. If not installed, the program gracefully handles the absence and displays an installation message.

## Reflection

### Challenges
1. **ABC Parsing Complexity:** Parsing ABC notation required handling multiple edge cases: accidentals, key signatures, octave modifiers, duration notation, and repeat bars. The most challenging aspect was correctly mapping ABC note names to MIDI note numbers while respecting key signatures and local accidentals.

2. **MIDI Timing:** MIDI messages use relative time (delta time), meaning each message's time value is relative to the previous message. Handling rests required accumulating silent time and adding it to the next note-on event, which took careful tracking.

3. **Noise Generation:** Implementing pink and brown noise required algorithms (Voss-McCartney for pink noise, cumulative sum for brown) that generate perceptually correct frequency characteristics.

4. **Audio Synchronization:** When mixing external WAV files, ensuring proper resampling, channel handling (mono conversion), and duration matching required careful audio processing.

### What I Learned
1. **Digital Signal Processing:** Gained deep understanding of waveform generation, ADSR envelopes, and audio normalization techniques.

2. **ABC Notation:** Learned the intricacies of ABC notation, including key signature handling, accidentals scoping, and timing notation.

3. **MIDI Format:** Understood MIDI file structure, including delta time encoding, meta messages, and note event sequencing.

4. **Python Audio Libraries:** Worked with numpy for signal processing, wave module for WAV I/O, and mido for MIDI export. Learned graceful handling of optional dependencies.

5. **Code Organization:** Structured the code with clear separation between parsing, synthesis, and UI logic, making it maintainable and extensible.

## Testing Summary

### Test Cases Performed
1. **ABC Parsing:**
   - Tested basic note parsing (A–G, a–g)
   - Verified octave modifiers (', comma)
   - Tested accidentals (^, _, =)
   - Validated key signature application (C, G, F, etc.)
   - Confirmed duration parsing (numbers, fractions, slashes)
   - Tested rest handling (z)
   - Verified repeat bars (|: :|)

2. **Audio Synthesis:**
   - Tested all four waveforms (sine, square, sawtooth, triangle)
   - Verified ADSR envelope application
   - Tested BPM changes and tempo accuracy
   - Validated pitch shifting (± semitones)

3. **Noise and Mixing:**
   - Tested white, pink, and brown noise generation
   - Verified noise envelope application
   - Tested external WAV mixing with various file formats

4. **Export Functions:**
   - Tested WAV export and playback
   - Verified MIDI export with various BPMs and pitch shifts
   - Confirmed MIDI velocity mapping from loudness
   - Tested graceful handling when mido is not installed

5. **Menu Functionality:**
   - Tested all 11 menu options
   - Verified input validation and error handling
   - Confirmed settings persistence across operations

### Test Files
- `test.abc`: Contains a demo tune with various note types, accidentals, and rests for comprehensive testing

## AI Usage Declaration

See Appendix B for detailed AI usage declaration.

## References

1. ABC Notation Standard: https://abcnotation.com/wiki/abc:standard:v2.1
2. MIDI Specification: Standard MIDI File Format
3. Voss-McCartney Algorithm for Pink Noise Generation
4. NumPy Documentation: https://numpy.org/doc/
5. Mido Library Documentation: https://mido.readthedocs.io/


