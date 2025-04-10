"""
High-level interface to vamp plugins

Depends on `vamphost <https://pypi.org/project/vamphost/>`_ and on some vamp plugins
being installed (pyin plugin)

These dependencies should be taken care of automatically when installing maelzel
(the pyin plugin is shipped together with maelzel, the vamphost module is provided
via pypi and installed by pip as a dependency)

"""
from __future__ import annotations

import os
from dataclasses import dataclass
from math import isnan
from typing import TYPE_CHECKING

import bpf4
import bpf4.core
import bpf4.util
import numpy as np
import numpyx
import pitchtools as pt
import vamp
import vamp.frames
import vamp.load
import vamp.process
from emlib import iterlib

from maelzel import stats
from maelzel._util import getPlatform
from maelzel.common import getLogger
from maelzel.snd import numpysnd

logger = getLogger('maelzel.snd')


if TYPE_CHECKING:
    from typing import Set


_pyinThresholdDistrs = {
    "uniform": 0,
    "beta10": 1,
    "beta15": 2,
    "beta30": 3,
    "single10": 4,
    "single15": 5,
    "single20": 7
}

_cache = {}


class Note:
    """A Note represents an event in time with duration and frequency"""

    __slots__ = ('timestamp', 'frequency', 'duration')

    def __init__(self, timestamp: float, frequency: float, duration: float):
        self.timestamp = timestamp
        """The start time"""

        self.frequency = frequency
        """The frequency in Hz"""

        self.duration = duration
        """The duration of the event"""


def vampFolder(pluginbits=64) -> str:
    """
    Returns the user vamp plugins folder

    This is the folder where the user should place vamp plugins. In linux, vamp
    plugins installed via a package manager (apt-get, for example), place the vamp
    plugins under ``/usr/...``. This folder is reserved for the system and not returned
    here.

    This function does not ensure the existence of the returned directory.

    Args:
        pluginbits: the architecture of the plugin, if known. One of
            32 or 64, as int. This is only relevant for windows

    Returns:
        the installation folder where the vamp plugin should be placed


    https://www.vamp-plugins.org/download.html?platform=linux64&search=key&go=Go


    ================   =========================================================
    Operating System   Plugin folder
    ================   =========================================================
    macos              ``$HOME/Library/Audio/Plug-Ins/Vamp``
    windows 64bits     64-bit plugins in ``C:/Program Files/Vamp Plugins``
                       32-bit plugins in ``C:/Program Files (x86)/Vamp Plugins``
                       Both 32- and 64-bit plugins can be used, as long as they
                       are placed in the correct folder.
    windows 32bits     ``C:/Program Files/Vamp Plugins``
                       64 bit plugins cannot be used
    linux              ~/vamp
                       Only plugins with the correct architecture can be used
    ================   =========================================================
    """
    osname, arch = getPlatform()
    if osname == 'linux':
        return os.path.expanduser('~/vamp')
    elif osname == 'darwin':
        return os.path.expanduser('~/Library/Audio/Plug-Ins/Vamp')
    elif osname == 'windows':
        if pluginbits == 32 or not pluginbits:
            return r'C:\Program Files (x86)\Vamp Plugins'
        else:
            return r'C:\Program Files\Vamp Plugins'
    else:
        raise RuntimeError(f"Platform {osname}:{arch} not supported")


def listPlugins(cached=True) -> Set[str]:
    """
    List all available vamp plugins

    Args:
        cached: if True, use cache when querying multiple times

    Returns:
        A set of plugin identifiers
    """
    if cached:
        plugins = _cache.get('plugins')
        if plugins is not None:
            return plugins
    else:
        import importlib
        importlib.reload(vamp.vampyhost)

    _cache['plugins'] = plugins = set(vamp.list_plugins())
    return plugins


def pyinAvailable() -> bool:
    """Is the pyin plugin available?"""
    return 'pyin:pyin' in listPlugins()


def pyinNotes(samples: np.ndarray,
              sr: int,
              fftSize=2048,
              stepSize=256,
              lowAmpSuppression=0.01,
              threshDistr="beta15",
              onsetSensitivity=0.9,
              pruneThresh=0.1
              ) -> list[Note]:
    """
    Notes detection. Uses pyin

    pYIN vamp plugin: https://code.soundsoftware.ac.uk/projects/pyin/files

    Args:
        samples: the sample data (mono)
        sr: the sample-rate
        fftSize: fft size
        stepSize: hop size in samples
        lowAmpSuppression: supress low amplitude pitch estimates.
            As a reference, 0.01 = -40dB, 0.001 = -60dB
        threshDistr: yin threshold distribution. See table 1 below
        onsetSensitivity: onset sensitivity
        pruneThresh: totalDuration pruning threshold

    Returns:
        a list of Notes, where each Note has the attributes .timestamp, .frequency, .totalDuration

    ============   ============
    thresh_distr   Description
    ============   ============
    uniform        Uniform
    beta10         Beta (mean 0.10)
    beta15         Beta (mean 0.15)
    beta30         Beta (mean 0.30)
    single10       Single value 0.10
    single15       Single value 0.15
    single20       Single value 0.20
    ============   ============
    """
    assert 'pyin:pyin' in listPlugins(), \
        "Vamp plugin 'pyin' not found. Install it from https://code.soundsoftware.ac.uk/projects/pyin"

    if fftSize < 2048:
        raise ValueError("The pyin vamp plugin does not accept fft size less than 2048")

    if len(samples.shape) > 1:
        samples = samples[:,0]
    threshdistridx = _pyinThresholdDistrs.get(threshDistr)
    if threshdistridx is None:
        raise ValueError(f"Unknown threshold distribution: {threshDistr}. "
                         f"It must be one of {', '.join(_pyinThresholdDistrs.keys())}")

    output_unvoiced = "negative"
    output_unvoiced_idx = {
        False: 0,
        True: 1,
        "negative": 2
    }.get(output_unvoiced)
    if output_unvoiced_idx is None:
        raise ValueError(f"Unknown output_unvoiced value {output_unvoiced}. "
                         f"possible values: {False, True, 'negative'}")

    params = {
        'lowampsuppression': lowAmpSuppression,
        'onsetsensitivity': onsetSensitivity,
        'prunethresh': pruneThresh,
        'threshdistr': threshdistridx,
        'outputunvoiced': output_unvoiced_idx
    }
    result = vamp.collect(data=samples, sample_rate=sr, plugin_key="pyin:pyin",
                          output="notes", block_size=fftSize,
                          step_size=stepSize, parameters=params)
    notes = []
    for onset in result['list']:
        n = Note(onset['timestamp'].to_float(), float(onset['values'][0]), onset['totalDuration'].to_float())
        notes.append(n)
    return notes


@dataclass
class PyinResult:
    voicedProbabilityCurve: bpf4.core.Linear
    f0candidates: list[tuple[float, list[float], list[float]]]
    """A list of tuples (timestamp, f0candidates, f0candidateprobability)"""

    smoothPitchCurve: bpf4.core.Linear
    """Smooth pitch curve"""

    smoothPitchCurveNan: bpf4.core.Linear
    """Like smoothPitchCurve, but unvoiced sections are marked with 'nan' values"""

    f0curve: bpf4.core.Linear
    """F0 curve resulting from picking the best f0 candidate over time. nan where there is no candidate"""

    voicedProbabilityQuantile: stats.Quantile1d

    rmsCurve: bpf4.core.BpfInterface
    """bpf mapping rms in time"""

    rmsDbQuantile: stats.Quantile1d
    """rms quantile evaluator in dB, maps db values to their quantile"""

    numCandidates: bpf4.core.BpfInterface

    def __repr__(self):
        def _(cs, maxnum=4, fmt='.6g'):
            if cs is None:
                return 'None'
            if len(cs) > maxnum:
                end = ', …]'
                cs = cs[:maxnum]
            else:
                end = ']'
            return '[' + ', '.join(format(x, fmt) for x in cs) + end
        maxframes = 10
        if len(self.f0candidates) > maxframes:
            end = ', ...]'
        else:
            end = ']'
        frames = [f"({t:.3f}s, {_(c, fmt='.1f')}, {_(p, fmt='.4f')})" for t, c, p in self.f0candidates[:maxframes]]
        framestr = '[' + ', '.join(frames) + end
        return (f"PyinResult(voicedProbabilityCurve={self.voicedProbabilityCurve}, smoothPitchCurve={self.smoothPitchCurve}, "
                f"f0curve={self.f0curve}, f0candidates={framestr}")


def pyin(samples: np.ndarray,
         sr: int,
         fftSize=2048,
         overlap=8,
         lowAmpSuppressionDb=-60,
         lowAmpSuppressionPercentile=0.01,
         threshDistr='beta15',
         onsetSensitivity=0.7,
         pruneThresh=0.1,
         voicedThresholdPercentile=0.1,
         preciseTime=False,
         minRmsPercentile=0.05,
         rmsPeriod=0.020,
         maxRelativeSkew=0.15
         ) -> PyinResult:
    """
    Pyin analysis, enhanced with some extra features

    Args:
        samples: the samples to analyze
        sr: samplerate
        fftSize: FFT size
        overlap: determines hopsize = fftSize // overlap
        lowAmpSuppressionDb: low amplitude suppression in dB
        threshDistr:
        onsetSensitivity:
        pruneThresh:
        voicedThresholdPercentile:
        preciseTime:
        minRmsPercentile:
        rmsPeriod:
        maxRelativeSkew:

    Returns:

    """
    # r = vamptools.pyin(samples2, sr, overlap=8, voicedThresholdPercentile=0.1, onsetSensitivity=0.1, threshDistr='beta15', lowAmpSuppression=0.01, preciseTime=True, rmsPeriod=0.020, minRmsPercentile=0.002)
    if fftSize < 2048:
        raise ValueError("The pyin vamp plugin does not accept fft size less than 2048")

    if len(samples.shape) > 1:
        samples = samples[:,0]
    threshdistridx = _pyinThresholdDistrs.get(threshDistr)
    if threshdistridx is None:
        raise ValueError(f"Unknown threshold distribution: {threshDistr}. "
                         f"It must be one of {', '.join(_pyinThresholdDistrs.keys())}")

    output_unvoiced_idx = 2  # 0=no, 1=yes, 2=yes, with negative freq

    step_size = fftSize // overlap
    kwargs = {'step_size': step_size, "block_size": fftSize}
    plugin_key = "pyin:pyin"
    params = {
        'lowampsuppression': pt.db2amp(lowAmpSuppressionDb),
        'onsetsensitivity': onsetSensitivity,
        'prunethresh': pruneThresh,
        'threshdistr': threshdistridx,
        'outputunvoiced': output_unvoiced_idx,
        'precisetime': int(preciseTime)
    }
    plugin, step_size, block_size = vamp.load.load_and_configure(samples, sr, plugin_key,
                                                                 params, **kwargs)

    ff = vamp.frames.frames_from_array(samples, step_size, block_size)
    outputs = ['smoothedpitchtrack', 'voicedprob', 'f0candidates', 'f0probs']
    results = list(vamp.process.process_with_initialised_plugin(ff,
                                                                sample_rate=sr,
                                                                step_size=step_size,
                                                                plugin=plugin,
                                                                outputs=outputs))

    vps = [d['voicedprob'] for d in results if 'voicedprob' in d]
    pts = [d['smoothedpitchtrack'] for d in results if 'smoothedpitchtrack' in d]
    f0s = [d['f0candidates'] for d in results if 'f0candidates' in d]
    f0probs = [d['f0probs'] for d in results if 'f0probs' in d]

    vptimes = [vp['timestamp'].to_float() for vp in vps]
    vpvalues = [float(vp['values'][0]) for vp in vps]
    vpcurve = bpf4.core.Linear(vptimes, vpvalues)
    voicedProbabilityQuantile = stats.Quantile1d(vpvalues)
    voicedThreshold = voicedProbabilityQuantile.value(voicedThresholdPercentile)

    smoothpitchtimes = [frame['timestamp'].to_float() for frame in pts]

    for t0, t1 in iterlib.pairwise(smoothpitchtimes):
        if t1 < t0:
            raise RuntimeError(f"times not sorted, {t0=}, {t1=}")
        # elif t0 == t1:
            # raise RuntimeError(f"Duplicate times: {t0=}, {t1=}")

    smoothpitchfreqs = [float(frame['values'][0]) for frame in pts]
    smoothpitch = bpf4.core.Linear(smoothpitchtimes, smoothpitchfreqs)
    nanfreqs = []
    nantimes = []
    for t, f in zip(smoothpitchtimes, smoothpitchfreqs):
        if f < 0:
            if not nanfreqs or (nanfreqs and not isnan(nanfreqs[-1])):
                nantimes.append(t)
                nanfreqs.append(float('nan'))
        else:
            nantimes.append(t)
            nanfreqs.append(f)

    smoothpitchnan = bpf4.core.Linear(nantimes, nanfreqs)

    rmscurve0 = numpysnd.rmsBpf(samples, sr=sr, dt=rmsPeriod, overlap=2)
    rmscurve = bpf4.util.smoothen(rmscurve0, window=rmsPeriod*4)
    rmsnum = rmscurve0.dxton(rmsPeriod)
    rmsdbquant = stats.Quantile1d(rmscurve0.amp2db().map(rmsnum))
    silencermsdb = rmsdbquant.value(minRmsPercentile)
    silencerms = pt.db2amp(silencermsdb)
    # silencerms = rmshist.percentileToValue(minRmsPercentile)
    f0candidates = []
    for f0sframe, f0probsframe in zip(f0s, f0probs):
        t = f0sframe['timestamp'].to_float()
        assert t == f0probsframe['timestamp'].to_float()
        values = f0sframe.get('values')
        voiced = vpcurve(t)
        smoothfreq = smoothpitch(t)
        if values is not None and voiced > voicedThreshold and smoothfreq > 0 and rmscurve(t) >= silencerms:
            row = (t, values, f0probsframe['values'])
            f0candidates.append(row)
        else:
            f0candidates.append((t, None, None))

    times = [t for t, _, _ in f0candidates]
    lencandidates = [float(len(candidates)) if candidates is not None else 0 for _, candidates, _ in f0candidates]
    numCandidatesInTime = bpf4.core.NoInterpol(times, lencandidates)
    # numCandidates = bpf4.core.NoInterpol(*zip(*[(t, len(candidates) if candidates is not None else 0)
      #                                           for t, candidates, probs in f0candidates]))  # type: ignore

    f0pairs: list[tuple[float, float]] = []
    for t, candidates, probabilities in f0candidates:
        if candidates is not None:
            best = candidates[0]
            f0smooth = smoothpitch(t)
            if f0pairs and f0smooth > 0 and abs(best - f0smooth) / f0pairs[-1][1] > maxRelativeSkew:
                best = f0smooth
            f0pairs.append((t, best))
        else:
            if f0pairs and f0pairs[-1][1] > 0:
                # Only add a nan if the last breakpoint was not a nan
                f0pairs.append((t, float('nan')))
    xs, ys = zip(*f0pairs)
    f0bestcurve = bpf4.core.Linear(np.array(xs), np.array(ys))

    return PyinResult(voicedProbabilityCurve=vpcurve,
                      f0candidates=f0candidates,
                      smoothPitchCurve=smoothpitch,
                      f0curve=f0bestcurve,
                      smoothPitchCurveNan=smoothpitchnan,
                      voicedProbabilityQuantile=voicedProbabilityQuantile,
                      rmsCurve=rmscurve,
                      rmsDbQuantile=rmsdbquant,
                      numCandidates=numCandidatesInTime)


def pyinPitchTrack(samples: np.ndarray,
                   sr: int,
                   fftSize=2048,
                   overlap=8,
                   lowAmpSuppression=0.01,
                   threshDistr="beta15",
                   onsetSensitivity=0.7,
                   pruneThresh=0.1,
                   outputUnvoiced='negative'
                   ) -> np.ndarray:
    """
    Analyze the samples and extract fundamental and voicedness

    For each measurement calculates the fundamental frequency and the voicedness
    probability  (the confidence that sound is pitched at a given time).

    pYIN vamp plugin: https://code.soundsoftware.ac.uk/projects/pyin/files

    Args:
        samples: the audio samples (mono). If a multichannel sample
            is given, only the first channel will be processed
        sr: sample rate
        fftSize: fft size (vamp names this "block_size"). Must be >= 2048
        overlap: determines the hop size (hop size = fftSize // overlap)
        lowAmpSuppression: supress low amplitude pitch estimates. 0.01=-40dB, 0.001=-60dB
        threshDistr: yin threshold distribution. See table 1 below
        onsetSensitivity: onset sensitivity
        pruneThresh: totalDuration pruning threshold
        outputUnvoiced: method used to output frequencies when the sound is
            unvoiced (there is no reliable pitch detected). Choices are True (sets
            the frequency to 'nan' for unvoiced breakpoints), False (the breakpoint
            is skipped) or 'negative' (outputs the detected frequency as negative)


    Returns:
        a 2D numpy array (float32) of 3 column with one row for each step.
        The columns are: time, f0, voiced probability (~= confidence)

        Whenever the confidence that the f0 is correct drops below
        a certain threshold, the frequency is given as negative

    ============   ============
    thresh_distr   Description
    ============   ============
    uniform        Uniform
    beta10         Beta (mean 0.10)
    beta15         Beta (mean 0.15)
    beta30         Beta (mean 0.30)
    single10       Single value 0.10
    single15       Single value 0.15
    single20       Single value 0.20
    ============   ============

    Example::

        import sndfileio
        import csoundengine as ce
        from from maelzel.snd.vamptools import *
        samples, sr = sndfileio.sndread("/path/to/soundfile.wav")
        matrix = pyin_pitchtrack(samples, sr)
        times = matrix[:,0]
        freqs = matrix[:,1]
        TODO
        freqbpf = bpf.core.Linear(times, matrix[:,1])
        midibpf = freqbpf.f2m()[::0.05]
        voicedprob = bpf.core.Linear(times, matrix[:,2])
        # play both the sample and the f0 to check
        tabnum = ce.makeTable(samples, sr=sr, block=True)
        ce.playSample(tabnum)
        synth = ce.session().sched('.sine', pargs={'kmidi':pitch(0)})
        synth.automatePargs('kmidi', pitch.flat_pairs())
        synth.automatePargs('kamp', pitchpitch.flat_pairs())

        TODO
    """
    assert 'pyin:pyin' in listPlugins(), \
        "Vamp plugin 'pyin' not found. Install it from https://code.soundsoftware.ac.uk/projects/pyin"

    if fftSize < 2048:
        raise ValueError("The pyin vamp plugin does not accept fft size less than 2048")

    if len(samples.shape) > 1:
        samples = samples[:,0]
    threshdistridx = _pyinThresholdDistrs.get(threshDistr)
    if threshdistridx is None:
        raise ValueError(f"Unknown threshold distribution: {threshDistr}. "
                         f"It must be one of {', '.join(_pyinThresholdDistrs.keys())}")

    if isinstance(outputUnvoiced, bool):
        logger.warning("bool values are deprecated. Use 'nan' instead of True and 'skip' instead of False")

    output_unvoiced_idx = {
        'skip': 0,
        False: 0,
        'nan': 1,
        True: 1,
        "negative": 2
    }.get(outputUnvoiced)

    if output_unvoiced_idx is None:
        raise ValueError(f"outputUnvoiced should be 'nan', 'skip' or 'negative', got {outputUnvoiced}")

    step_size = fftSize // overlap
    kwargs = {'step_size': step_size, "block_size": fftSize}
    plugin_key = "pyin:pyin"
    params = {
        'lowampsuppression':lowAmpSuppression,
        'onsetsensitivity':onsetSensitivity,
        'prunethresh':pruneThresh,
        'threshdistr':threshdistridx,
        'outputunvoiced':output_unvoiced_idx
    }
    plugin, step_size, block_size = vamp.load.load_and_configure(samples, sr, plugin_key,
                                                                 params, **kwargs)

    ff = vamp.frames.frames_from_array(samples, step_size, block_size)
    outputs = ['smoothedpitchtrack', 'voicedprob', 'f0candidates', 'f0probs']
    results = list(vamp.process.process_with_initialised_plugin(ff,
                                                                sample_rate=sr,
                                                                step_size=step_size,
                                                                plugin=plugin,
                                                                outputs=outputs))

    vps = [d['voicedprob'] for d in results if 'voicedprob' in d]
    pts = [d['smoothedpitchtrack'] for d in results if 'smoothedpitchtrack' in d]
    f0s = [d['f0candidates'] for d in results if 'f0candidates' in d]

    arr = np.empty((len(vps), 3))
    i = 0
    NAN = float('nan')
    for vp, track, f0 in zip(vps, pts, f0s):
        t = vp['timestamp']
        probs = vp['values']
        candidates = f0.get('values', None)
        freq = float(track['values'][0])
        if freq < 0:
            if outputUnvoiced == 'nan':
                freq = NAN
            prob = 0
        elif candidates is None:
            prob = 0
        else:
            candidates = candidates.astype('float64')
            if len(candidates) == len(probs):
                idx = numpyx.nearestidx(candidates, freq, sorted=False)
                prob = probs[idx]
            else:
                prob = probs[0]
        arr[i] = [t, freq, prob]
        i += 1
    return arr


def pyinSmoothPitch(samples: np.ndarray,
                    sr:int,
                    fftSize=2048,
                    stepSize=256,
                    lowAmpSuppression=0.01,
                    threshDistr="beta15",
                    onsetSensitivity=0.7,
                    outputUnvoiced='nan',
                    pruneThresh=0.1) -> tuple[float, np.ndarray]:
    """
    Fundamental frequency analysis

    Args:
        samples (np.ndarray): the audio samples (mono). If a multichannel sample
            is given, only the first channel will be processed
        sr (int): sample rate
        fftSize (int): fft size (vamp names this "block_size")
        stepSize (int): hop size in samples
        lowAmpSuppression (float): supress low amplitude pitch estimates.
          As a reference, 0.01 = -40dB, 0.001 = -60dB
        threshDistr (str): yin threshold distribution. See table 1 below
        onsetSensitivity (float): onset sensitivity
        pruneThresh (float): totalDuration pruning threshold
        outputUnvoiced: method used to output frequencies when the sound is
            unvoiced (there is no reliable pitch detected). Choices are 'nan', (sets
            the frequency to 'nan' for unvoiced breakpoints), 'skip' (the breakpoint
            is skipped) or 'negative' (outputs the detected frequency as negative)

    Returns:
        a tuple (delta time, frequencies: np.ndarray)

    Table 1
    ~~~~~~~

    ============   ============
    thresh_distr   Description
    ============   ============
    uniform        Uniform
    beta10         Beta (mean 0.10)
    beta15         Beta (mean 0.15)
    beta30         Beta (mean 0.30)
    single10       Single value 0.10
    single15       Single value 0.15
    single20       Single value 0.20
    ============   ============
    """
    if 'pyin:pyin' not in listPlugins():
        raise RuntimeError(f"Vamp plugin 'pyin' not found. Installed plugins: {listPlugins()}. "
                           f"Install pyin from https://code.soundsoftware.ac.uk/projects/pyin")

    if fftSize < 2048:
        raise ValueError("The pyin vamp plugin does not accept fft size less than 2048")

    if len(samples.shape) > 1:
        samples = samples[:,0]
    threshdistridx = _pyinThresholdDistrs.get(threshDistr)
    if threshdistridx is None:
        raise ValueError(f"Unknown threshold distribution: {threshDistr}. "
                         f"It must be one of {', '.join(_pyinThresholdDistrs.keys())}")

    if isinstance(outputUnvoiced, bool):
        logger.warning("bool values are deprecated. Use 'nan' instead of True and 'skip' instead of False")
    outputUnvoicedIndex = {
        'skip': 0,
        False: 0,
        'nan': 1,
        True: 1,
        "negative": 2
    }.get(outputUnvoiced)

    if outputUnvoicedIndex is None:
        raise ValueError(f"outputUnvoiced should be 'nan', 'skip' or 'negative', got {outputUnvoiced}")

    params = {
        'lowampsuppression': lowAmpSuppression,
        'onsetsensitivity': onsetSensitivity,
        'prunethresh': pruneThresh,
        'threshdistr': threshdistridx,
        'outputunvoiced': outputUnvoicedIndex
    }
    result1 = vamp.collect(data=samples, sample_rate=sr, plugin_key="pyin:pyin",
                           output="smoothedpitchtrack", block_size=fftSize,
                           step_size=stepSize, parameters=params)
    dt, freqs = result1['vector']
    freqsarray = np.array(freqs, dtype=float)
    return (float(dt), freqsarray)
