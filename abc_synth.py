#!/usr/bin/env python3
"""
abc_synth.py — Menu-driven ABC synthesizer

Usage
-----
Run: python3 abc_synth.py

Menu Options:
1) Select waveform: sine | square | sawtooth | triangle
2) Set loudness (0.0–1.0 master gain)
3) Indicate ABC file path
4) Change speed (BPM)
5) Shift pitch (semitones)
6) Add background noise (white/pink/brown) with optional envelope
7) Mix external WAV
8) Play
9) Save as WAV
10) Exit
11) Save as MIDI (requires mido library)

Dependencies:
- Required: numpy
- Optional playback: simpleaudio
- Optional MIDI export: mido

Install:
python3 -m pip install --user numpy simpleaudio mido

Features
--------
1) Select waveform: sine | square | sawtooth | triangle
2) Set loudness (0.0–1.0 master gain)
3) Indicate ABC file path
4) Change speed (BPM)
5) Shift pitch (semitones)
6) Add background noise (white/pink/brown) with optional envelope
7) Mix external WAV
8) Play
9) Save as WAV
10) Exit
11) Save as MIDI

ABC support (required + some extras)
------------------------------------
Headers: X, T, K, C, M, Q   (L: supported if present; default 1/8)
Body   : A–G/a–g notes, commas and apostrophes for octaves, numbers, /, rest z,
         bar lines |, ||; accidentals ^ = _; repeat bars |: :| (extra credit)

ADSR envelope is applied to every synthesized note.

Tested with Python 3.9+. Requires numpy. Playback uses simpleaudio if installed.
MIDI export requires mido library.
"""

from __future__ import annotations
import math
import os
import re
import sys
import wave
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np

# Optional playback backends
_SIMPLEAUDIO_OK = False
try:
    import simpleaudio as sa
    _SIMPLEAUDIO_OK = True
except Exception:
    _SIMPLEAUDIO_OK = False

# Optional MIDI export
_MIDO_OK = False
try:
    import mido
    _MIDO_OK = True
except Exception:
    _MIDO_OK = False

_IS_WINDOWS = sys.platform.startswith("win")
if _IS_WINDOWS:
    try:
        import winsound
    except Exception:
        winsound = None


# ----------------------- Audio / DSP helpers -----------------------

def db_to_lin(db: float) -> float:
    return 10 ** (db / 20.0)

def lin_to_db(lin: float) -> float:
    lin = max(lin, 1e-12)
    return 20 * math.log10(lin)

def normalize(sig: np.ndarray, peak: float = 0.98) -> np.ndarray:
    maxv = np.max(np.abs(sig)) if sig.size else 1.0
    if maxv < 1e-12:
        return sig
    return (sig / maxv) * peak

def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or x.size == 0:
        return x.copy()
    duration = x.size / src_sr
    t_src = np.linspace(0, duration, x.size, endpoint=False)
    n_dst = int(round(duration * dst_sr))
    t_dst = np.linspace(0, duration, n_dst, endpoint=False)
    return np.interp(t_dst, t_src, x).astype(np.float32)

def pink_noise(n: int) -> np.ndarray:
    # Voss–McCartney algorithm (approx 1/f)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    rows = 16
    array = np.empty((rows, n), dtype=np.float32)
    array[0] = np.random.randn(n)
    for i in range(1, rows):
        step = 2 ** i
        vals = np.random.randn((n + step - 1) // step).repeat(step)[:n]
        array[i] = vals
    pink = array.sum(axis=0)
    pink /= np.max(np.abs(pink)) + 1e-12
    return pink.astype(np.float32)

def brown_noise(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    white = np.random.randn(n).astype(np.float32)
    brown = np.cumsum(white)
    brown -= brown.mean()
    brown /= np.max(np.abs(brown)) + 1e-12
    return brown.astype(np.float32)


# ----------------------- Waveform Generators -----------------------

def osc_sine(phase: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * phase)

def osc_square(phase: np.ndarray) -> np.ndarray:
    return np.sign(osc_sine(phase))

def osc_saw(phase: np.ndarray) -> np.ndarray:
    # Range [-1,1)
    return 2.0 * (phase - np.floor(phase + 0.5))

def osc_triangle(phase: np.ndarray) -> np.ndarray:
    return 2.0 * np.abs(osc_saw(phase)) - 1.0


WAVEFORMS = {
    "sine": osc_sine,
    "square": osc_square,
    "sawtooth": osc_saw,
    "triangle": osc_triangle,
}


# ----------------------- ADSR Envelope -----------------------

@dataclass
class ADSR:
    attack: float = 0.01
    decay: float = 0.05
    sustain: float = 0.8
    release: float = 0.08

    def apply(self, signal: np.ndarray, sr: int) -> np.ndarray:
        n = signal.size
        if n == 0:
            return signal
        a = max(int(self.attack * sr), 1)
        d = max(int(self.decay * sr), 1)
        r = max(int(self.release * sr), 1)
        s = max(n - (a + d + r), 0)

        env = np.empty(n, dtype=np.float32)
        idx = 0
        # Attack
        if a > 0:
            env[idx:idx + a] = np.linspace(0, 1, a, endpoint=False)
            idx += a
        # Decay
        if d > 0:
            env[idx:idx + d] = np.linspace(1, self.sustain, d, endpoint=False)
            idx += d
        # Sustain
        if s > 0:
            env[idx:idx + s] = self.sustain
            idx += s
        # Release
        remain = n - idx
        if remain > 0:
            env[idx:] = np.linspace(self.sustain, 0, remain, endpoint=True)
        return (signal * env).astype(np.float32)


# ----------------------- ABC Parsing -----------------------

NOTE_RE = re.compile(
    r"""
    (?P<bar>\|\:|\:\||\|\||\|)|
    (?P<rest>z)|
    (?P<note>
        (?P<acc>[\^_=]{1,2})?
        (?P<name>[A-Ga-g])
        (?P<oct>[',]*)
        (?P<dur>(\d+)?(\/\d+|\/)?)
    )
    """,
    re.VERBOSE
)

HEADER_RE = re.compile(r'^\s*([A-Za-z]):\s*(.+?)\s*$')

# Basic key signature maps (major only) for accidentals; minor will be treated as relative major
KEY_SIGNATURES = {
    "C": [],
    "G": ["F#"],
    "D": ["F#", "C#"],
    "A": ["F#", "C#", "G#"],
    "E": ["F#", "C#", "G#", "D#"],
    "B": ["F#", "C#", "G#", "D#", "A#"],
    "F#": ["F#", "C#", "G#", "D#", "A#", "E#"],
    "C#": ["F#", "C#", "G#", "D#", "A#", "E#", "B#"],
    "F": ["Bb"],
    "Bb": ["Bb", "Eb"],
    "Eb": ["Bb", "Eb", "Ab"],
    "Ab": ["Bb", "Eb", "Ab", "Db"],
    "Db": ["Bb", "Eb", "Ab", "Db", "Gb"],
    "Gb": ["Bb", "Eb", "Ab", "Db", "Gb", "Cb"],
    "Cb": ["Bb", "Eb", "Ab", "Db", "Gb", "Cb", "Fb"],
}

NOTE_TO_SEMITONE = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}

@dataclass
class ABCHeader:
    X: Optional[str] = None
    T: Optional[str] = None
    K: str = "C"
    C: Optional[str] = None
    M: Optional[str] = None
    Q: int = 120  # BPM
    L: Tuple[int, int] = (1, 8)  # default note length 1/8 if provided; ABC default is 1/8 if no L

@dataclass
class ABCNote:
    midi: Optional[int]  # None for rest
    duration_beats: float

@dataclass
class ABCScore:
    header: ABCHeader
    notes: List[ABCNote] = field(default_factory=list)


def parse_header(lines: List[str]) -> Tuple[ABCHeader, int]:
    hdr = ABCHeader()
    i = 0
    while i < len(lines):
        m = HEADER_RE.match(lines[i])
        if not m:
            break
        tag, val = m.group(1).upper(), m.group(2).strip()
        if tag == 'X': hdr.X = val
        elif tag == 'T': hdr.T = val
        elif tag == 'K': hdr.K = val
        elif tag == 'C': hdr.C = val
        elif tag == 'M': hdr.M = val
        elif tag == 'Q':
            bpm = re.findall(r'\d+', val)
            if bpm:
                hdr.Q = int(bpm[-1])
        elif tag == 'L':
            frac = val.strip()
            if '/' in frac:
                num, den = frac.split('/', 1)
                hdr.L = (int(num or '1'), int(den or '8'))
        i += 1
    return hdr, i


def key_signature_accidentals(key: str) -> Dict[str, int]:
    # Return default accidental for names A..G in this key: -1 flat, +1 sharp, 0 natural
    base = key.strip()
    is_minor = base.endswith('m') or 'min' in base.lower()
    if is_minor:
        # naive: map minor to relative major (up 3 semitones)
        tonic = base[:-1] if base.endswith('m') else base
        # not robust; fall back to A minor (no sharps/flats) if unknown
        rel_major = "C"
        order = ["C","G","D","A","E","B","F#","C#","F","Bb","Eb","Ab","Db","Gb","Cb"]
        if tonic.title() == "A": rel_major = "C"
        elif tonic.title() == "E": rel_major = "G"
        elif tonic.title() == "B": rel_major = "D"
        elif tonic.title() == "F#": rel_major = "A"
        elif tonic.title() == "C#": rel_major = "E"
        elif tonic.title() == "G#": rel_major = "B"
        elif tonic.title() == "D#": rel_major = "F#"
        elif tonic.title() == "A#": rel_major = "C#"
        elif tonic.title() == "D": rel_major = "F"
        elif tonic.title() == "G": rel_major = "Bb"
        elif tonic.title() == "C": rel_major = "Eb"
        elif tonic.title() == "F": rel_major = "Ab"
        elif tonic.title() == "Bb": rel_major = "Db"
        elif tonic.title() == "Eb": rel_major = "Gb"
        elif tonic.title() == "Ab": rel_major = "Cb"
        base = rel_major
    accs = {n:0 for n in "ABCDEFG"}
    for s in KEY_SIGNATURES.get(base.title(), []):
        n = s[0]
        sign = 1 if s.endswith('#') else -1
        accs[n] = sign
    return accs


def abc_duration_to_beats(token: str, default_len: Tuple[int,int]) -> float:
    """ token examples: '', '2', '3/2', '/', '//', '/4' """
    if not token:
        num, den = default_len
        return num/den
    # numeric part then optional slash part
    m = re.fullmatch(r'(\d+)?(\/(\d+)?)?', token)
    if not m:
        num, den = default_len
        return num/den
    whole = m.group(1)
    slash = m.group(2)
    if whole and not slash:
        mult = int(whole)
        num, den = default_len
        return mult * (num/den)
    if slash:
        if whole:
            # '3/2' style
            den = m.group(3)
            den = int(den) if den else 2
            return int(whole)/den
        else:
            # '/' or '/4'
            den = m.group(3)
            den = int(den) if den else 2
            return 1/den
    num, den = default_len
    return num/den


def note_name_octave_to_midi(name: str, acc: int, octave_marks: str) -> int:
    # ABC: uppercase = octave starting at middle C? Convention:
    # We'll treat:
    #   C  = MIDI 60 (middle C) when name == 'C' and with no octave marks? Actually in ABC:
    #   'c' (lowercase) = middle C (MIDI 60). Uppercase C = C one octave below (MIDI 48).
    # So:
    #   upper 'C' => octave 4 below 'c' base. We'll set:
    #   base for lowercase 'c' = 60
    base_midi_c_lower = 60  # 'c'
    if name.islower():
        # 'c'..'b': octave with middle C at 60
        base_oct_offset = 0
        pitch_class = NOTE_TO_SEMITONE[name.upper()]
        midi = base_midi_c_lower + pitch_class
    else:
        # 'C'..'B' : one octave below
        pitch_class = NOTE_TO_SEMITONE[name]
        midi = base_midi_c_lower - 12 + pitch_class

    # Apply octave marks: "'" raises by 12 per mark; "," lowers by 12 per mark (for uppercase base)
    if octave_marks:
        up = octave_marks.count("'")
        down = octave_marks.count(",")
        midi += 12 * (up - down)

    # Apply accidental ( -2 .. +2 )
    midi += acc
    return midi


def parse_body(body: str, header: ABCHeader) -> List[ABCNote]:
    acc_defaults = key_signature_accidentals(header.K)
    default_len = header.L
    notes: List[ABCNote] = []

    # Local accidentals reset at each bar
    local_acc: Dict[Tuple[str,int], int] = {}

    pos = 0
    for m in NOTE_RE.finditer(body):
        if m.group('bar'):
            # Reset local accidentals at bar line; handle repeats (extra credit)
            token = m.group('bar')
            if token == '|:':
                # mark repeat start
                repeat_start = len(notes)
            elif token == ':|' and 'repeat_start' in locals():
                # repeat notes from repeat_start to end
                segment = notes[repeat_start:]
                notes.extend(segment)
                del repeat_start
            local_acc.clear()
            continue

        if m.group('rest'):
            dur = abc_duration_to_beats('', default_len)
            dtoken = ''
            # attempt to get duration part from regex by re-parsing a tiny window after 'z'
            span_end = m.end()
            # Not perfect; but we already captured 'note' variant separately. For rest, allow immediate digits or slashes
            # We'll look ahead in text (simple).
            # (For simplicity, we ignore here; you can encode rests with explicit duration via zN in source.)
            notes.append(ABCNote(midi=None, duration_beats=dur))
            continue

        if m.group('note'):
            acc_str = m.group('acc') or ''
            name = m.group('name')
            octv = m.group('oct') or ''
            durtok = m.group('dur') or ''
            # accidental parsing
            acc = 0
            if acc_str:
                # ^^ => +2, ^ => +1, = => 0, _ => -1, __ => -2
                if acc_str == '^^': acc = +2
                elif acc_str == '^': acc = +1
                elif acc_str == '__': acc = -2
                elif acc_str == '_': acc = -1
                elif acc_str == '=': acc = 0
            else:
                # apply key signature default if no local accidental override
                base_name = name.upper()
                acc = acc_defaults.get(base_name, 0)

            midi = note_name_octave_to_midi(name, acc, octv)
            dur_beats = abc_duration_to_beats(durtok, default_len)
            notes.append(ABCNote(midi=midi, duration_beats=dur_beats))
            continue

    return notes


def parse_abc_text(text: str) -> ABCScore:
    lines = [ln.rstrip('\n') for ln in text.splitlines() if ln.strip() != '']
    header, i = parse_header(lines)
    body = "\n".join(lines[i:])
    notes = parse_body(body, header)
    return ABCScore(header=header, notes=notes)


# ----------------------- Synthesizer -----------------------

@dataclass
class SynthSettings:
    sr: int = 44100
    waveform: str = "sine"
    loudness: float = 0.8
    bpm: int = 120
    pitch_shift_semitones: int = 0
    adsr: ADSR = field(default_factory=ADSR)
    noise_type: Optional[str] = None  # 'white'|'pink'|'brown'|None
    noise_level: float = 0.0          # 0..1 relative to signal peak
    noise_envelope: bool = False
    mix_wav_path: Optional[str] = None
    mix_level: float = 0.5            # 0..1

class Synthesizer:
    def __init__(self, settings: SynthSettings):
        self.s = settings

    def note_freq_from_midi(self, midi: int) -> float:
        midi_shifted = midi + self.s.pitch_shift_semitones
        return 440.0 * (2.0 ** ((midi_shifted - 69) / 12.0))

    def render_note(self, midi: Optional[int], beats: float) -> np.ndarray:
        sec = (60.0 / max(self.s.bpm, 1)) * beats
        n = max(int(round(sec * self.s.sr)), 1)
        if midi is None:
            sig = np.zeros(n, dtype=np.float32)
            return self.s.adsr.apply(sig, self.s.sr)

        freq = self.note_freq_from_midi(midi)
        t = np.arange(n, dtype=np.float32) / self.s.sr
        phase = (freq * t) % 1.0
        osc = WAVEFORMS.get(self.s.waveform, osc_sine)
        sig = osc(phase).astype(np.float32)
        sig = self.s.adsr.apply(sig, self.s.sr)
        return sig

    def render_score(self, score: ABCScore) -> np.ndarray:
        # adopt BPM from header unless overridden already by menu
        if score.header.Q and self.s.bpm != score.header.Q:
            # We keep the menu's bpm; if you want to honor the file's Q, set the menu to 'use file'
            pass
        chunks = [self.render_note(n.midi, n.duration_beats) for n in score.notes]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(chunks)
        # Add noise
        if self.s.noise_type and self.s.noise_level > 0.0:
            n = audio.size
            if self.s.noise_type == 'white':
                noise = np.random.randn(n).astype(np.float32)
            elif self.s.noise_type == 'pink':
                noise = pink_noise(n)
            elif self.s.noise_type == 'brown':
                noise = brown_noise(n)
            else:
                noise = np.zeros(n, dtype=np.float32)

            if self.s.noise_envelope:
                # simple triangle envelope 0 -> 1 -> 0 over full duration
                env = np.concatenate([
                    np.linspace(0, 1, max(1, n//2), endpoint=False),
                    np.linspace(1, 0, n - max(1, n//2), endpoint=True)
                ]).astype(np.float32)
                noise *= env

            noise = normalize(noise, peak=1.0)
            audio = audio + self.s.noise_level * noise

        # Mix with external WAV if requested
        if self.s.mix_wav_path and os.path.isfile(self.s.mix_wav_path):
            mix_sig, mix_sr = read_wav_mono(self.s.mix_wav_path)
            mix_sig = resample_linear(mix_sig, mix_sr, self.s.sr)
            # pad/trim to match
            if mix_sig.size < audio.size:
                mix_sig = np.pad(mix_sig, (0, audio.size - mix_sig.size))
            else:
                mix_sig = mix_sig[:audio.size]
            audio = (1.0 - self.s.mix_level) * audio + self.s.mix_level * mix_sig

        # master gain and safety normalization
        audio = normalize(audio, peak=1.0) * float(self.s.loudness)
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        return audio


# ----------------------- WAV I/O and Playback -----------------------

def save_wav(path: str, signal: np.ndarray, sr: int = 44100):
    sig16 = np.clip(signal, -1.0, 1.0)
    sig16 = (sig16 * 32767.0).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(sig16.tobytes())

def read_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        nframes = wf.getnframes()
        data = wf.readframes(nframes)
        dtype = np.int16 if wf.getsampwidth() == 2 else None
        if dtype is None:
            raise ValueError("Only 16-bit PCM WAV supported for mixing.")
        x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if ch == 2:
            x = x.reshape(-1, 2).mean(axis=1)
        return x, sr

def play_audio(signal: np.ndarray, sr: int = 44100):
    if signal.size == 0:
        print("Nothing to play.")
        return
    if _SIMPLEAUDIO_OK:
        audio16 = (np.clip(signal, -1, 1) * 32767).astype(np.int16)
        play_obj = sa.play_buffer(audio16, 1, 2, sr)
        play_obj.wait_done()
    elif _IS_WINDOWS and winsound is not None:
        # quick temp wav
        tmp = "_tmp_play.wav"
        save_wav(tmp, signal, sr)
        winsound.PlaySound(tmp, winsound.SND_FILENAME)
        try:
            os.remove(tmp)
        except Exception:
            pass
    else:
        print("Playback requires `simpleaudio` (pip install simpleaudio). Saving still works.")


# ----------------------- MIDI Export -----------------------

def save_midi(path: str, score: ABCScore, bpm: int, pitch_shift_semitones: int, loudness: float):
    """
    Save ABC score as MIDI file.
    
    Args:
        path: Output file path (will append .mid if missing)
        score: Parsed ABC score
        bpm: Tempo in BPM
        pitch_shift_semitones: Pitch shift to apply
        loudness: Master gain (0-1), mapped to MIDI velocity (20-110)
    """
    if not _MIDO_OK:
        print("MIDI export requires 'mido'. Install with: python3 -m pip install --user mido")
        return
    
    # Ensure .mid extension
    if not path.lower().endswith('.mid'):
        path += '.mid'
    
    # Create MIDI file with one track
    ticks_per_beat = 480
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Add tempo meta message (microseconds per quarter note)
    microseconds_per_beat = int(60000000 / bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
    
    # Map loudness (0-1) to MIDI velocity (20-110)
    velocity = int(20 + (loudness * 90))
    velocity = max(20, min(110, velocity))
    
    accumulated_time = 0
    
    for note in score.notes:
        # Convert duration from beats to ticks
        duration_ticks = int(round(note.duration_beats * ticks_per_beat))
        
        if note.midi is not None:
            # Apply pitch shift
            midi_note = note.midi + pitch_shift_semitones
            # Clamp to valid MIDI range (0-127)
            midi_note = max(0, min(127, midi_note))
            
            # Note on (time relative to last message - accumulated time from rests)
            track.append(mido.Message('note_on', note=midi_note, velocity=velocity, time=accumulated_time))
            # Note off after duration
            track.append(mido.Message('note_off', note=midi_note, velocity=velocity, time=duration_ticks))
            accumulated_time = 0  # Reset for next note
        else:
            # Rest: accumulate silence (will be added to next note_on time)
            accumulated_time += duration_ticks
    
    # Save file
    mid.save(path)
    print(f"Saved: {path}")


# ----------------------- Menu App -----------------------

@dataclass
class AppState:
    settings: SynthSettings = field(default_factory=SynthSettings)
    abc_path: Optional[str] = None
    last_audio: Optional[np.ndarray] = None
    last_score: Optional[ABCScore] = None

class MenuApp:
    def __init__(self):
        self.state = AppState()

    def run(self):
        while True:
            self.show_menu()
            choice = input("Select option (1-11): ").strip()
            handlers = {
                '1': self.menu_waveform,
                '2': self.menu_loudness,
                '3': self.menu_abc_path,
                '4': self.menu_bpm,
                '5': self.menu_pitch_shift,
                '6': self.menu_noise,
                '7': self.menu_mix_wav,
                '8': self.menu_play,
                '9': self.menu_save,
                '10': self.menu_exit,
                '11': self.menu_save_midi,
            }
            if choice in handlers:
                try:
                    handlers[choice]()
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Invalid option.")

    def show_menu(self):
        s = self.state.settings
        print("\n=== ABC Synth Menu ===")
        print(f"1- Selecting a waveform           [current: {s.waveform}]")
        print(f"2- Setting the loudness           [current: {s.loudness:.2f}]")
        print(f"3- Indicating the ABC file path   [current: {self.state.abc_path or 'not set'}]")
        print(f"4- Changing speed (BPM)           [current: {s.bpm}]")
        print(f"5- Shifting pitch (semitones)     [current: {s.pitch_shift_semitones}]")
        noise_str = f"{s.noise_type or 'off'} lvl={s.noise_level:.2f} env={'on' if s.noise_envelope else 'off'}"
        print(f"6- Adding background noise        [current: {noise_str}]")
        mix_str = self.state.settings.mix_wav_path or 'off'
        print(f"7- Mixing within an external WAV  [current: {mix_str} (level {s.mix_level:.2f})]")
        print(f"8- Playing the file")
        print(f"9- Saving the music as a WAV file")
        print(f"10- Exit")
        print(f"11- Saving the music as a MIDI file")

    # ---- menu handlers ----
    def menu_waveform(self):
        print("Choose waveform: sine | square | sawtooth | triangle")
        wf = input("Waveform: ").strip().lower()
        if wf not in WAVEFORMS:
            print("Invalid waveform.")
            return
        self.state.settings.waveform = wf
        print(f"Waveform set to {wf}.")

    def menu_loudness(self):
        try:
            val = float(input("Set loudness (0.0-1.0): ").strip())
            if not (0.0 <= val <= 1.0):
                raise ValueError()
            self.state.settings.loudness = val
            print(f"Loudness set to {val:.2f}.")
        except Exception:
            print("Invalid loudness.")

    def menu_abc_path(self):
        path = input("Enter ABC file path: ").strip().strip('"')
        if not os.path.isfile(path):
            print("File not found.")
            return
        self.state.abc_path = path
        # Load & parse immediately to give feedback
        score = self.load_score()
        if score:
            print(f"Loaded: {score.header.T or 'Untitled'} (BPM {score.header.Q}, Key {score.header.K})")

    def menu_bpm(self):
        try:
            bpm = int(input("Set BPM (e.g., 120): ").strip())
            if bpm <= 0:
                raise ValueError()
            self.state.settings.bpm = bpm
            print(f"BPM set to {bpm}.")
        except Exception:
            print("Invalid BPM.")

    def menu_pitch_shift(self):
        try:
            st = int(input("Shift in semitones (e.g., -2, 0, +5): ").strip())
            self.state.settings.pitch_shift_semitones = st
            print(f"Pitch shift set to {st} semitones.")
        except Exception:
            print("Invalid value.")

    def menu_noise(self):
        typ = input("Noise type (off | white | pink | brown): ").strip().lower()
        if typ == 'off':
            self.state.settings.noise_type = None
            self.state.settings.noise_level = 0.0
            print("Noise disabled.")
            return
        if typ not in ('white','pink','brown'):
            print("Invalid type.")
            return
        try:
            lvl = float(input("Noise level 0.0-1.0 (e.g., 0.15): ").strip())
            env = input("Apply envelope to noise? (y/n): ").strip().lower().startswith('y')
        except Exception:
            print("Invalid values.")
            return
        self.state.settings.noise_type = typ
        self.state.settings.noise_level = max(0.0, min(1.0, lvl))
        self.state.settings.noise_envelope = bool(env)
        print(f"Noise set: {typ} @ {self.state.settings.noise_level:.2f}, envelope={'on' if env else 'off'}.")

    def menu_mix_wav(self):
        path = input("External WAV path (or 'off'): ").strip().strip('"')
        if path.lower() == 'off':
            self.state.settings.mix_wav_path = None
            print("External mix disabled.")
            return
        if not os.path.isfile(path):
            print("File not found.")
            return
        try:
            lvl = float(input("Mix level 0.0-1.0 (how much external wav): ").strip())
        except Exception:
            print("Invalid mix level.")
            return
        self.state.settings.mix_wav_path = path
        self.state.settings.mix_level = max(0.0, min(1.0, lvl))
        print(f"External WAV set. Level {self.state.settings.mix_level:.2f}")

    def ensure_audio(self) -> Optional[np.ndarray]:
        if self.state.last_audio is not None:
            return self.state.last_audio
        score = self.load_score()
        if score is None:
            return None
        synth = Synthesizer(self.state.settings)
        audio = synth.render_score(score)
        self.state.last_audio = audio
        self.state.last_score = score
        return audio

    def load_score(self) -> Optional[ABCScore]:
        if not self.state.abc_path:
            print("Set ABC file path first.")
            return None
        with open(self.state.abc_path, 'r', encoding='utf-8') as f:
            text = f.read()
        score = parse_abc_text(text)
        # If user wants file's Q to drive bpm, uncomment the next line:
        # self.state.settings.bpm = score.header.Q
        return score

    def menu_play(self):
        # recompute every time to reflect setting changes
        self.state.last_audio = None
        audio = self.ensure_audio()
        if audio is None:
            return
        play_audio(audio, self.state.settings.sr)

    def menu_save(self):
        # recompute with current settings
        self.state.last_audio = None
        audio = self.ensure_audio()
        if audio is None:
            return
        out = input("Output WAV path (e.g., output.wav): ").strip().strip('"')
        if not out.lower().endswith('.wav'):
            out += '.wav'
        save_wav(out, audio, self.state.settings.sr)
        print(f"Saved: {out}")

    def menu_save_midi(self):
        score = self.load_score()
        if score is None:
            return
        out = input("Save MIDI file as (e.g., output.mid): ").strip().strip('"')
        save_midi(out, score, self.state.settings.bpm, 
                 self.state.settings.pitch_shift_semitones, 
                 self.state.settings.loudness)

    def menu_exit(self):
        print("Goodbye!")
        sys.exit(0)


# ----------------------- Main -----------------------

def main():
    app = MenuApp()
    app.run()

if __name__ == "__main__":
    main()