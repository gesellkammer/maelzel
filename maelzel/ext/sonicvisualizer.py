"""
Utilities to interact with Sonic Visualizer
"""
from __future__ import annotations
import os
import bpf4 as bpf
from pitchtools import *
from emlib.containers import RecordList
from emlib import csvtools

from typing import Union


def readNotes(path: str) -> RecordList:
    """
    Reads the data exported by Export Annotation from the layer 'Notes' / 'Flexible Notes'

    Returns:
        a RecordList with fields 'start', 'freq', 'dur', 'amp', 'label'

    The format is:

    - one row per note
    - columns: time, freq, dur, amp, label

    """
    ext = os.path.splitext(path)[1].lower()
    if ext != '.csv':
        raise ValueError(f"format {ext} not supported")
    data = csvtools.readcsv(path, columns=["start", "freq", "dur", "amp", "label"])
    notes = [f2n(d.freq) for d in data]
    return data.add_column('note', notes)


def readQtrans(path: str, minpitch=36, octaveDivision=12):
    """
    Read a Q-Transform analysis

    Args:
        path: 
        minpitch: 
        octaveDivision: 

    Returns:
        a QTransf
    """
    from lxml import etree
    t = etree.parse(path)
    root = t.getroot()
    data = root._find("data")
    model = data._find("model")
    dset = data._find("dataset")
    assert model.get("name").split(":")[-1].strip() == "Constant-Q Spectrogram"
    sr = int(model.get("sampleRate"))
    start = int(model.get("start"))
    end = int(model.get("end"))
    wsize = int(model.get("windowSize"))
    startframe = int(model.get("startFrame"))
    bins_per_row = int(model.get("yBinCount"))
    rows = []
    for row in dset.findall("row"):
        text = row.text
        if text:
            values = list(map(float, text.split()))
        else:
            values = []
        rows.append((int(row.get('n')), values))
    rows.sort(key=lambda r:r[0])
    values = list(zip(*rows))[1]   # only the column with the values
    return QTransform(start, wsize / sr, values, minpitch=minpitch,
                      octave_division=octaveDivision)


class QTransform:
    def __init__(self, start, dt, values, minpitch, octave_division):
        self.start = start
        self.dt = dt
        self.values = values
        self.minpitch = minpitch
        self.octave_division = octave_division
        self.max_idx = len(values)
        self.maxpitch = minpitch + len(values[0]) * (12 / octave_division)
    
    def __call__(self, t: float, midi: Union[float, str]) -> float:
        if isinstance(midi, str):
            midi = n2m(midi)
        idx = int((t - self.start) / self.dt)
        midi_idx = int((midi - self.minpitch) * (self.octave_division / 12.))
        value = self.values[idx][midi_idx]
        return value
    
    def chordAt(self, t: float, maxnotes=8, minamp=-60) -> list[tuple[str, float]]:
        idx = int((t - self.start) / self.dt)
        values = self.values[idx]
        dp = 12. / self.octave_division
        pitches = [self.minpitch + i * dp for i in range(len(values))]
        notes = list(zip(pitches, values))
        notes2 = sorted(notes, key=lambda n:n[1], reverse=True)
        notes2 = [(m2n(n), v) for n, v in notes2]
        minamp = db2amp(minamp)
        if maxnotes is not None:
            notes2 = notes2[:maxnotes]
        notes3 = [n for n in notes2 if n[1] >= minamp]
        return notes3


def readAdaptiveSpectr(path: str) -> Spectrum:
    t = etree.parse(path)
    root = t.getroot()
    data = root._find("data")
    model = data._find("model")
    dset = data._find("dataset")
    bins = []
    for bin in dset.findall("bin"):
        freq = int(bin.get("name").split()[0])
        bins.append(freq)
    rows = []
    for row in dset.findall("row"):
        text = row.text
        if text:
            values = list(map(float, text.split()))
        else:
            values = []
        rows.append((int(row.get('n')), values))
    rows.sort(key=lambda r:r[0])
    values = list(zip(*rows))[1]   # only the column with the values
    sr = int(model.get("sampleRate"))
    start = int(model.get("start"))
    wsize = int(model.get("windowSize"))
    return Spectrum(bins, values, start, wsize/sr)


class Spectrum:
    def __init__(self, freqs, values, start, dt):
        self.freqs = freqs
        self.values = values
        self.start = start
        self.dt = dt
        bpfs = []
        for amps in values:
            if amps:
                bpfs.append(bpf.core.NoInterpol(freqs, amps))
            else:
                bpfs.append(bpf.const(0))
        self.bpfs = bpfs

