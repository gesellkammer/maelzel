"""
Utilities to interact with Sonic Visualizer
"""
import os
from lxml import etree
from bpf4 import bpf
from emlib.pitchtools import *


def readnotes(path):
    """
    Reads the data as exported by Export Annotation
    from the layer 'Notes' or Flexible Notes'

    Returns a pandas DataFrame

    The format is:

    - one row per note
    - columns: time, freq, dur, amp, label

    Use dataframe2chord to convert it to a list of rows of 
    type: (pitch, amp, notename)
    """
    import pandas

    def readcsv(path):
        data = pandas.read_csv(path, names="start freq dur amp label".split())
        notes = list(map(f2n, data['freq']))
        data['note'] = notes
        return data
    
    ext = os.path.splitext(path)[1].lower()

    if ext == '.csv':
        return readcsv(path)
    else:
        raise ValueError("format %s not supported" % ext)


def read_qtransf(path, minpitch=36, octave_division=12):
    t = etree.parse(path)
    root = t.getroot()
    data = root.find("data")
    model = data.find("model")
    dset = data.find("dataset")
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
    values = zip(*rows)[1]  # only the column with the values
    return QTransf(start, wsize / sr, values, minpitch=minpitch, octave_division=octave_division)


class QTransf(object):
    def __init__(self, start, dt, values, minpitch, octave_division):
        self.start = start
        self.dt = dt
        self.values = values
        self.minpitch = minpitch
        self.octave_division = octave_division
        self.max_idx = len(values)
        self.maxpitch = minpitch + len(values[0]) * (12 / octave_division)
    
    def __call__(self, t, midi):
        if isinstance(midi, six.string_types):
            midi = n2m(midi)
        idx = int((t - self.start) / self.dt)
        midi_idx = int((midi - self.minpitch) * (self.octave_division / 12.))
        value = self.values[idx][midi_idx]
        return value
    
    def chord_at(self, t, maxnotes=8, minamp=-60):
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


def read_adaptive_spectr(path):
    t = etree.parse(path)
    root = t.getroot()
    data = root.find("data")
    model = data.find("model")
    dset = data.find("dataset")
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
    values = zip(*rows)[1]  # only the column with the values
    sr = int(model.get("sampleRate"))
    start = int(model.get("start"))
    end = int(model.get("end"))
    wsize = int(model.get("windowSize"))
    return Spectrum(bins, values, start, wsize/sr)


def dataframe2chord(df):
    chord = []
    for i, row in df.iterrows():
        pitch = f2m(row['freq'])
        amp = row['amp']
        notename = m2n(pitch)
        chord.append((pitch, amp, notename))
    chord.sort(reverse=True)
    return chord


class Spectrum(object):
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

