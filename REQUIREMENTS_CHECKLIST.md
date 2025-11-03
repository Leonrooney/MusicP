# ABC Synthesizer - Requirements Checklist

## ✅ Core Features (Menu Options 1-10)

### Menu Option 1 - Waveform Selection
- [x] Supports sine waveform
- [x] Supports square waveform
- [x] Supports sawtooth waveform
- [x] Supports triangle waveform
- [x] Menu displays current waveform selection

### Menu Option 2 - Loudness Control
- [x] Accepts loudness values 0.0-1.0
- [x] Master gain applied to audio output
- [x] Menu displays current loudness value

### Menu Option 3 - ABC File Path
- [x] User can specify ABC file path
- [x] Validates file existence
- [x] Loads and parses ABC file on input
- [x] Displays loaded tune information (title, BPM, key)
- [x] Menu displays current ABC file path

### Menu Option 4 - BPM (Speed)
- [x] User can set BPM
- [x] Validates BPM > 0
- [x] BPM affects note durations
- [x] Menu displays current BPM

### Menu Option 5 - Pitch Shift
- [x] User can shift pitch in semitones
- [x] Accepts positive and negative values
- [x] Pitch shift applied to synthesized notes
- [x] Menu displays current pitch shift

### Menu Option 6 - Background Noise
- [x] Supports white noise
- [x] Supports pink noise
- [x] Supports brown noise
- [x] Noise can be disabled (off)
- [x] User can set noise level (0.0-1.0)
- [x] Optional envelope can be applied to noise
- [x] Menu displays current noise settings

### Menu Option 7 - External WAV Mixing
- [x] User can specify external WAV file path
- [x] Validates WAV file existence
- [x] Mixes external WAV with synthesized audio
- [x] User can set mix level (0.0-1.0)
- [x] Can disable external mixing (off)
- [x] Menu displays current mix settings

### Menu Option 8 - Play Audio
- [x] Plays synthesized audio
- [x] Works with simpleaudio if installed
- [x] Gracefully handles missing simpleaudio
- [x] Recomputes audio with current settings

### Menu Option 9 - Save as WAV
- [x] User can specify output file path
- [x] Automatically appends .wav extension if missing
- [x] Saves audio as WAV file
- [x] Recomputes audio with current settings
- [x] Confirms successful save

### Menu Option 10 - Exit
- [x] Exits program gracefully
- [x] Displays goodbye message

## ✅ Extra Credit Feature - Menu Option 11 (MIDI Export)

### MIDI Export Implementation
- [x] Menu option 11 added: "Saving the music as a MIDI file"
- [x] Uses `mido` library if available
- [x] Gracefully handles missing `mido` library
- [x] Displays installation message if `mido` not installed
- [x] Prompts user for output file name
- [x] Automatically appends `.mid` extension if missing

### MIDI File Format
- [x] Creates one MIDI track
- [x] Uses ticks_per_beat = 480
- [x] Adds tempo meta message based on BPM setting
- [x] Converts note durations from beats to ticks (beats × ticks_per_beat)
- [x] Maps loudness (0-1) to MIDI velocity (20-110)
- [x] Applies pitch shift in semitones to note numbers
- [x] Handles rests as silent time (accumulated delay to next note)
- [x] Clamps MIDI note numbers to valid range (0-127)

### MIDI Export Requirements
- [x] Does not affect WAV export functionality
- [x] Does not affect playback functionality
- [x] Works independently from other features

## ✅ ABC Parsing Support

### Headers
- [x] X: Reference number
- [x] T: Title
- [x] K: Key signature
- [x] C: Composer
- [x] M: Meter (e.g., 4/4, 3/4)
- [x] Q: Tempo in BPM
- [x] L: Default note length

### Body Notation
- [x] Notes: A-G (uppercase) and a-g (lowercase)
- [x] Octave modifiers: commas (,) and apostrophes (')
- [x] Durations: numbers, fractions (e.g., /2, 3/2)
- [x] Rests: z
- [x] Bar lines: | (single), || (double)
- [x] Accidentals: ^ (sharp), _ (flat), = (natural)
- [x] Double accidentals: ^^ (double sharp), __ (double flat)

### Extra Credit - Repeat Bars
- [x] Start repeat: |:
- [x] End repeat: :|
- [x] Automatically plays repeated sections

### ADSR Envelope
- [x] Applied to every synthesized note
- [x] Attack phase
- [x] Decay phase
- [x] Sustain phase
- [x] Release phase

## ✅ Code Quality & Documentation

### Docstring Updates
- [x] Added "Usage" section to docstring
- [x] Lists all menu options (1-11)
- [x] Lists dependencies (numpy, simpleaudio, mido)
- [x] Includes installation instructions
- [x] Updated Python version requirement

### Menu Display
- [x] Shows options 1-11
- [x] Displays current settings for each option
- [x] Input prompt shows correct range (1-11)

### Handler Mapping
- [x] All options 1-11 mapped to handlers
- [x] Option 11 mapped to `menu_save_midi()` function
- [x] Error handling for invalid options

## ✅ File Requirements

### Read Me.txt
- [x] Specifies OS: macOS
- [x] Specifies IDE: VS Code / Terminal
- [x] Includes Python version (Python 3.9.6)
- [x] Lists required dependencies (numpy)
- [x] Lists optional dependencies (simpleaudio, mido)
- [x] Includes installation command
- [x] Includes run command
- [x] Includes notes about PATH for --user installs

### Report.md
- [x] Title section
- [x] Author(s) section
- [x] Student ID(s) section (placeholder)
- [x] Overview of program
- [x] ABC parsing and assumptions
- [x] Audio synthesis (waveforms, ADSR, BPM)
- [x] Noise and mixing
- [x] MIDI export section
- [x] Reflection (challenges, what was learned)
- [x] Testing summary
- [x] AI usage declaration (placeholder for Appendix B)
- [x] References section

### test.abc
- [x] Contains X:1 header
- [x] Contains T:Cursor Demo Tune title
- [x] Contains C:Anonymous composer
- [x] Contains M:4/4 meter
- [x] Contains L:1/8 default note length
- [x] Contains Q:120 tempo
- [x] Contains K:C key signature
- [x] Contains repeat bars |: :|
- [x] Contains various note types
- [x] Contains accidentals (^, =, _)
- [x] Contains rests (z)
- [x] Contains octave modifiers (c', G,)

## ✅ Functionality Tests

### Basic Operations
- [x] Program starts without errors
- [x] Menu displays correctly with all 11 options
- [x] Option 3 loads test.abc successfully
- [x] Option 8 plays audio (if simpleaudio installed)
- [x] Option 9 saves WAV file successfully
- [x] Option 11 handles MIDI export (with or without mido)

### Path Handling
- [x] No hard-coded absolute paths
- [x] Relative paths work correctly
- [x] File extensions appended automatically (.wav, .mid)

### Error Handling
- [x] Invalid menu options handled gracefully
- [x] Missing ABC files handled gracefully
- [x] Missing mido library handled gracefully
- [x] Missing simpleaudio handled gracefully
- [x] Invalid input values handled gracefully

## ✅ MIDI Export Technical Requirements

### MIDI Specifications Met
- [x] ticks_per_beat = 480 ✓
- [x] Tempo meta message from BPM ✓
- [x] Note durations: beats × ticks_per_beat ✓
- [x] Velocity: loudness mapped to 20-110 ✓
- [x] Pitch shift: applied to note numbers ✓
- [x] Rests: handled as silent time (accumulated delay) ✓
- [x] Single track format ✓

## Summary

**Total Requirements: 110+**
**Verified: ✅ All requirements met**

The ABC synthesizer meets all specified requirements including:
- All 11 menu options functional
- MIDI export feature fully implemented
- Comprehensive ABC parsing support
- All documentation files created
- Proper error handling and graceful degradation
- No hard-coded paths
- Clean code structure


