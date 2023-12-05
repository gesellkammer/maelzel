"""
Implements some helper functions to interact with audacity, in
particular to read markers and labels and convert 
them to useful representations in python

"""

from __future__ import annotations
import os
from pitchtools import f2n, f2m
from emlib.filetools import fixLineEndings
import bpf4 as bpf
from math import inf
from dataclasses import dataclass


@dataclass
class Label:
    start: float
    end: float
    label: str = ''


@dataclass
class Bin:
    freq: float
    level: float


@dataclass
class Note:
    note: str
    midi: float
    freq: float
    level: float
    step: int


def readLabels(filename: str) -> list[Label]:
    """
    import the labels generated in audacity

    Args:
        filename: the filename to read

    Returns:
        a list of labels
    """
    assert os.path.exists(filename)
    fixLineEndings(filename)
    f = open(filename, 'r')
    labels = []
    for line in f:
        words = line.split()
        if len(words) == 2:
            begin, end = words
            label = ''
        else:
            begin, end, label = words
        begin = float(begin)
        end = float(end)
        labels.append(Label(begin, end, label))
    return labels


def writeLabels(outfile: str, markers: list[tuple[float, float] | tuple[float, float, str]]
                ) -> None:
    """
    Write a labels file which can be imported by audacity.

    A label is a tuple if the form (start, end), or (start, end, name).
    If no name is given, an index number is used.

    Args:
        outfile: the file to write the labels to
        markers: a sequence of tuples (start, end) or (start, end, name)
    """
    labels = []
    for i, marker in enumerate(markers):
        if len(marker) == 2:
            start, end = marker
            name = str(i)
        elif len(marker) == 3:
            start, end, name = marker
            if not name:
                name = str(i)
        else:
            raise ValueError(
                "a Marker is a tuple of the form (start, end) or (start, end, label)")
        labels.append((start, end, name))
    if outfile is not None:
        with open(outfile, 'w') as f:
            for label in labels:
                f.write("\t".join(map(str, label)))
                f.write("\n")
    

def readSpectrum(path: str) -> list[Bin]:
    """
    Read a spectrum as saved by audacity

    Args:
        path: the path to the saved spectrum

    Returns:
        a list of Bins, where each bin is a tuple `(freq in Hz, level in dB)`
    """
    f = open(path)
    lines = f.readlines()[1:]
    out = []
    for line in lines:
        freq, level = list(map(float, line.split()))
        out.append(Bin(freq, level))
    return out


_dbToStepCurve = bpf.expon(
    -120, 0,
    -60, 0.0,
    -40, 0.1,
    -30, 0.4,
    -18, 0.9,
    -6, 1,
    0, 1,
    exp=0.3333333
)


def dbToStep(db: float, numsteps: int) -> int:
    """
    Convert dB value to histogram step
    """
    return int(_dbToStepCurve(db) * numsteps)


def readSpectrumAsChords(path: str, numsteps=8, maxNotesPerChord=inf
                         ) -> list[list[Note]]:
    """
    Reads the spectrum saved in `path` and splits it into at most `numsteps` chords

    The information saved by audacity represents the spectrum of the selected audio

    Args:
        path: the path of the saved spectrum (a .txt file)
        numsteps: the number of steps to split the spectral information into, according
            to their amplitude. Each step can be seen as a "layer"
        maxNotesPerChord: the max. number of bins for each "layer". Normally the loudest
            layers will have fewer components

    Returns:
        a list of chords, where each chord is a list of Note
    """
    data = readSpectrum(path)
    notes = [] 
    for bin_ in data:
        note = Note(note=f2n(bin_.freq), midi=f2m(bin_.freq), freq=bin_.freq, level=bin_.level,
                    step=dbToStep(bin_.level, numsteps))
        notes.append(note)
    chords = [[] for _ in range(numsteps)]
    notes2 = sorted(notes, key=lambda n:n.level, reverse=True)
    for note in notes2:
        chord = chords[note.step]
        if len(chord) <= maxNotesPerChord:
            chord.append(note)
    for chord in chords:
        chord.sort(key=lambda n:n.level, reverse=True)
    return chords


def readSpectrumAsBpf(path: str) -> bpf.BpfInterface:
    """
    Read the spectrum saved in `path`, returns a bpf mapping freq to level

    The information saved by audacity represents the spectrum of the selected audio
    """
    freqs = []
    levels = []
    f = open(path)
    lines = f.readlines()[1:]
    for line in lines:
        freq, level = list(map(float, line.split()))
        freqs.append(freq)
        levels.append(level)
    return bpf.core.Linear(freqs, levels)
