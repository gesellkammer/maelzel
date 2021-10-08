from __future__ import annotations
import numpy as np
import numpyx
from functools import cache
from typing import TYPE_CHECKING, NamedTuple
if TYPE_CHECKING:
    from typing import Tuple, List, Set

try:
    import vamp
    import vamp.frames
except ImportError:
    raise ImportError("vamp-host not installed. Install it via 'pip install vamphost'")


_pyin_threshold_distrs = {
    "uniform": 0,
    "beta10": 1,
    "beta15": 2,
    "beta30": 3,
    "single10": 4,
    "single15": 5,
    "single20": 7
}


class Note(NamedTuple):
    timestamp: float
    frequency: float
    duration: float


@cache
def list_plugins() -> Set[str]:
    """
    List all available vamp plugins

    Returns:
        A set of plugin identifiers
    """
    return set(vamp.list_plugins())


def pyin_notes(samples: np.ndarray, sr:int,
               fft_size=2048, step_size=256,
               low_amp_suppression=0.05,
               thresh_distr="beta15",
               onset_sensitivity=0.9,
               prune_thresh=0.1) -> List[Tuple[float, float, float]]:
    """
    Notes detection. Uses pyin

    pYIN vamp plugin: https://code.soundsoftware.ac.uk/projects/pyin/files

    Args:
        samples (np.ndarray): the sample data (mono)
        sr (int): the sample-rate
        fft_size (int): fft size
        step_size (int): hop size in samples
        low_amp_suppression (float): supress low amplitude pitch estimates
        thresh_distr (str): yin threshold distribution. See table 1 below
        onset_sensitivity (float): onset sensitivity
        prune_thresh (float): duration pruning threshold

    Returns:
        a list of Notes, where each Note is a namedtuple with attributes:
        (time, freq, duration)
    """
    if 'pyin:pyin' not in list_plugins():
        import webbrowser
        url = "https://code.soundsoftware.ac.uk/projects/pyin"
        downloadurl = "https://code.soundsoftware.ac.uk/projects/pyin/files"
        infourl = "https://www.vamp-plugins.org/download.html?platform=linux64&search=key&go=Go#install"
        webbrowser.open(url)
        webbrowser.open(infourl)
        print(f"Vamp plugin 'pyin' not found. Download it from {downloadurl}, "
              f"install it by following these instructions: {infourl}\n"
              f"Check the installation by calling list_plugins() and verify that"
              f"pyin:pyin is listed")
        raise RuntimeError("Plugin 'pyin' not found")

    if len(samples.shape) > 1:
        samples = samples[:,0]
    threshdistridx = _pyin_threshold_distrs.get(thresh_distr)
    if threshdistridx is None:
        raise ValueError(f"Unknown threshold distribution: {thresh_distr}. "
                         f"It must be one of {', '.join(_pyin_threshold_distrs.keys())}")

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
        'lowampsuppression': low_amp_suppression,
        'onsetsensitivity': onset_sensitivity,
        'prunethresh': prune_thresh,
        'threshdistr': threshdistridx,
        'outputunvoiced': output_unvoiced_idx
    }
    result = vamp.collect(data=samples, sample_rate=sr, plugin_key="pyin:pyin",
                          output="notes", block_size=fft_size,
                          step_size=step_size, parameters=params)
    notes = []
    for onset in result['list']:
        n = Note(onset['timestamp'], onset['values'][0], onset['duration'])
        notes.append(n)
    return notes


def pyin_pitchtrack(samples:np.ndarray, sr:int,
                    fft_size=2048, overlap=4,
                    low_amp_suppression=0.1,
                    thresh_distr="beta15",
                    onset_sensitivity=0.7,
                    prune_thresh=0.1
                    ) -> np.ndarray:
    """
    Analyze the samples and extract fundamental and voicedness

    For each measurement calculates the fundamental frequency and the voicedness
    probability  (the confidence that sound is pitched at a given time).

    pYIN vamp plugin: https://code.soundsoftware.ac.uk/projects/pyin/files

    **NB**: the returned array is of type float32

    Args:
        samples (np.ndarray): the audio samples (mono). If a multichannel sample
            is given, only the first channel will be processed
        sr (int): sample rate
        fft_size (int): fft size (vamp names this "block_size")
        overlap (int): determines the hop size (hop size = fft_size / overlap)
        low_amp_suppression (float): supress low amplitude pitch estimates
        thresh_distr (str): yin threshold distribution. See table 1 below
        onset_sensitivity (float): onset sensitivity
        prune_thresh (float): duration pruning threshold


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
    assert 'pyin:pyin' in list_plugins(), \
        "Vamp plugin 'pyin' not found. Install it from https://code.soundsoftware.ac.uk/projects/pyin"

    if len(samples.shape) > 1:
        samples = samples[:,0]
    threshdistridx = _pyin_threshold_distrs.get(thresh_distr)
    if threshdistridx is None:
        raise ValueError(f"Unknown threshold distribution: {thresh_distr}. "
                         f"It must be one of {', '.join(_pyin_threshold_distrs.keys())}")

    output_unvoiced = "negative"
    output_unvoiced_idx = {
        False: 0,
        True: 1,
        "negative": 2
    }.get(output_unvoiced)

    if output_unvoiced_idx is None:
        raise ValueError(f"Unknown output_unvoiced value {output_unvoiced}. "
                         f"possible values: {False, True, 'negative'}")
    step_size = fft_size // overlap
    kwargs = {'step_size': step_size, "block_size": fft_size}
    plugin_key = "pyin:pyin"
    params = {
        'lowampsuppression':low_amp_suppression,
        'onsetsensitivity':onset_sensitivity,
        'prunethresh':prune_thresh,
        'threshdistr':threshdistridx,
        'outputunvoiced':output_unvoiced_idx
    }
    plugin, step_size, block_size = vamp.load.load_and_configure(samples, sr, plugin_key,
                                                                 params, **kwargs)

    ff = vamp.frames.frames_from_array(samples, step_size, block_size)
    outputs = ['smoothedpitchtrack', 'voicedprob', 'f0candidates']
    results = list(vamp.process.process_with_initialised_plugin(
            ff, sr, step_size, plugin, outputs))

    vps = [d['voicedprob'] for d in results if 'voicedprob' in d]
    pts = [d['smoothedpitchtrack'] for d in results if 'smoothedpitchtrack' in d]
    f0s = [d['f0candidates'] for d in results if 'f0candidates' in d]

    arr = np.empty((len(vps), 3))
    i = 0
    for vp, pt, f0 in zip(vps, pts, f0s):
        t = vp['timestamp']
        probs = vp['values']
        candidates = f0.get('values', None)
        freq = float(pt['values'][0])
        if freq < 0 or candidates is None:
            prob = 0
        else:
            candidates = candidates.astype('float64')
            idx = numpyx.nearestidx(candidates, freq, sorted=False)
            prob = probs[idx]
            #pairs = zip(candidates, probs)
            #nearest = min(pairs, key=lambda pair:abs(pair[0]-freq))
            #prob = nearest[1]
        arr[i] = [t, freq, prob]
        i += 1
    return arr


def pyin_smoothpitch(samples: np.ndarray, sr:int,
                      fft_size=2048, step_size=256,
                      low_amp_suppression=0.1, thresh_distr="beta15",
                      onset_sensitivity=0.7,
                      prune_thresh=0.1) -> Tuple[float, np.ndarray]:
    """
    Fundamental frequency analysis

    Args:
        samples (np.ndarray): the audio samples (mono). If a multichannel sample
            is given, only the first channel will be processed
        sr (int): sample rate
        fft_size (int): fft size (vamp names this "block_size")
        step_size (int): hop size in samples
        low_amp_suppression (float): supress low amplitude pitch estimates
        thresh_distr (str): yin threshold distribution. See table 1 below
        onset_sensitivity (float): onset sensitivity
        prune_thresh (float): duration pruning threshold

    Returns:
        a tuple (delta time, frequencies: np.ndarray)

        Whenever the analysis determines that the sound is noise (unvoiced) without
        a clear fundamental, the frequency is given as negative.

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
    assert 'pyin:pyin' in list_plugins(), \
        "Vamp plugin 'pyin' not found. Install it from https://code.soundsoftware.ac.uk/projects/pyin"

    if len(samples.shape) > 1:
        samples = samples[:,0]
    threshdistridx = _pyin_threshold_distrs.get(thresh_distr)
    if threshdistridx is None:
        raise ValueError(f"Unknown threshold distribution: {thresh_distr}. "
                         f"It must be one of {', '.join(_pyin_threshold_distrs.keys())}")

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
        'lowampsuppression': low_amp_suppression,
        'onsetsensitivity': onset_sensitivity,
        'prunethresh': prune_thresh,
        'threshdistr': threshdistridx,
        'outputunvoiced': output_unvoiced_idx
    }
    result1 = vamp.collect(data=samples, sample_rate=sr, plugin_key="pyin:pyin",
                           output="smoothedpitchtrack", block_size=fft_size,
                           step_size=step_size, parameters=params)
    dt, freqs = result1['vector']
    return (dt, freqs)