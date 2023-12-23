"""
This module implements the Spectrum class

A Spectrum is the representation of a sound as a list of partials
in time. Normally these partials are the result of a partial-tracking
analysis.

This partial tracking representation can be used to manipulate the
sound (transpose in frequency, scale in time, quantize pitch, etc)
and can be also the basis for transcription in music notation.

"""
from __future__ import annotations
import numpy as np
import loristrck as lt
import bpf4
import pitchtools as pt
from emlib.filetools import normalizePath

from maelzel import histogram
from maelzel.snd import audiosample
from .partial import Partial
from . import pack


from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    import csoundengine


def _csoundEngine(name='maelzel') -> csoundengine.Engine:
    import csoundengine as ce
    return ce.getEngine(name) or ce.Engine(name=name)


def _firstPartialAfter(partials: list[Partial], t0: float) -> int:
    for i, p in enumerate(partials):
        if p.end > t0:
            return i
    return -1


def _partialsBetween(partials: list[Partial], t0=0., t1=0.) -> list[Partial]:
    """
    Return the partials present between t0 and t1

    Partials should be sorted

    This function is not optimized and performs a linear search over the
    partials. If this function is to be called repeatedly or within a
    performance relevant section, use `PartialIndex` instead

    Args:
        partials: a list of partials
        t0: start time in secs
        t1: end time in secs

    Returns:
        the partials within the time range (t0, t1)


    """
    if t1 == 0:
        t1 = max(p.end for p in partials)
    out = []
    for p in partials:
        if p.start > t1:
            break
        if p.end > t0:
            out.append(p)
    return out


class _PartialIndex:
    """
    Create an index to accelerate finding partials

    After creating the PartialIndex, each call to `partialindex.partials_between`
    should be faster than simply calling `partials_index` since the unoptimized
    function needs to always start a linear search from the beginning of the
    partials list.

    !!! note

        The index is only valid as long as the original partial list is not
        modified

    """
    def __init__(self, partials: list[Partial], end: float, dt=1.0):
        """
        Args:
            partials: the partials to index
            dt: the time resolution of the index. The lower this value the faster
                each query will be but the slower the creation of the index itself
        """
        self.start = partials[0].start
        self.end = end
        self.dt = dt
        self.partials = partials
        firstpartials = []
        startidx = 0
        for t in np.arange(self.start, end, dt):
            relidx = _firstPartialAfter(partials[max(0, startidx - 1):], float(t))
            absidx = startidx + relidx
            firstpartials.append(absidx)
            startidx = absidx
        self.firstpartials = firstpartials

    def partialsBetween(self, start: float, end: float) -> list[Partial]:
        """
        Returns the partials which are defined within the given time range

        Args:
            start: the start of the time interval
            end: the end of the time interval

        Returns:
            a list of partials present during the given time range
        """
        if start > end:
            raise ValueError(f"The start time should not be later than the end time, got {start=}, {end=}")
        idx = int((start - self.start) / self.dt)
        firstpartial = self.firstpartials[idx]
        if firstpartial < 0:
            return []
        return _partialsBetween(self.partials[firstpartial:], start, end)


class Spectrum:
    """
    A Spectrum represents a list of Partials

    A Spectrum is created from an analysis (see :func:`analyze`), read from a sdif file
    (see :meth:`Spectrum.read`) or from a selection of previously created Partials

    A Spectrum is considered immutable and any modification of its attributes is considered
    undefined behaviour

    Args:
        partials: the partials of this Spectrum
        indexTimeResolution: the time resolution used to index the partials. This is used internally
            to optimize searching for partials
    """

    def __init__(self, partials: list[Partial], indexTimeResolution=1.0):
        self.partials: list[Partial] = partials
        """The partials of this Spectrum"""

        self.start = self.partials[0].start if self.partials else 0
        """The start time of this spectrum"""

        self._end: float = 0.
        """The end time of this spectrum"""

        self._indexTimeResolution = indexTimeResolution
        """The time resolution used for indexing"""

        self._index: _PartialIndex | None = None
        self._packedMatrix: np.ndarray | None = None
        self._soundEngine: csoundengine.Engine | None = None

        if self.partials:
            self.partials.sort(key=lambda p: p.start)

    @property
    def end(self) -> float:
        """The end time of this spectrum"""
        if not self.partials:
            return 0.
        if self._end == 0:
            self._end = max(partial.end for partial in self.partials)
        return self._end

    @classmethod
    def read(cls, path: str) -> Spectrum:
        """
        Read a sdif file

        Args:
            path: the path to a sdif file (1TRC or RBEP)

        Returns:
            the Spectrum
        """
        arrays, labels = lt.read_sdif(path)
        return cls(partials=[Partial(array, label=label) for array, label in zip(arrays, labels)])

    def __iter__(self):
        return iter(self.partials)

    def __getitem__(self, item):
        out = self.partials.__getitem__(item)
        if isinstance(out, Partial):
            return out
        else:
            assert isinstance(out, list)
            return Spectrum(out)

    def __repr__(self):
        return f'Spectrum(numpartials={len(self)}, start={self.start:.3f}, end={self.end:.3f})'

    def __copy__(self):
        return Spectrum(self.partials.copy(), indexTimeResolution=self._indexTimeResolution)

    def __deepcopy__(self, memodict={}):
        return self.__copy__()

    def __add__(self, other):
        if isinstance(other, Spectrum):
            return Spectrum(self.partials + other.partials)

    def copy(self) -> Spectrum:
        """Copy this Spectrum"""
        return self.__deepcopy__()

    @property
    def index(self) -> _PartialIndex:
        if self._index is None:
            self._index = _PartialIndex(self.partials, end=self.end, dt=self._indexTimeResolution)
        return self._index

    def __len__(self):
        return len(self.partials)

    def partialsBetween(self, start: float, end: float, crop=False, minfreq=0, maxfreq=0
                        ) -> list[Partial]:
        """
        Returns a list of Partials which are present between the given times

        Partials listed here might start before `start` or end after `end`. Use crop
        or crop the partials directly to strictly limit the returned partials to the
        given time interval. Notice that cropping will potentially result in new partials
        being created since a cropped partial is a copy of the original (partials themselves
        are considered to be read-only)

        Args:
            start: the start time
            end: the end time
            crop: if True, partials are cropped to strictly fit between the given
                time interval

        Returns:
            a list of partials defined between the given time interval

        """
        selected = self.index.partialsBetween(start, end)
        if minfreq > 0 or maxfreq > 0:
            maxfreq = maxfreq or 24000
            selected = [p for p in selected if minfreq <= p.meanfreq() <= maxfreq]

        if not crop:
            return selected

        cropped = [p2 for p in selected
                   if (p2 := p.crop(start, end)) is not None]
        return cropped

    def write(self, outfile: str, rbep=True) -> None:
        """
        Write this Spectrum

        Formats supported: only ``.sdif`` at the moment

        The RBEP sdif format (with `rbep=True`) avoids resampling the spectrum and is the
        recommended way to save as sdif.

        Args:
            outfile: the path to write to (should end with .sdif)
            rbep: if True, use the RBEP format, otherwise the 1TRC format is used.

        """
        arrays = [p.data for p in self.partials]
        labels = [p.label for p in self.partials]
        outfile = normalizePath(outfile)
        lt.util.write_sdif(arrays, outfile=outfile, fmt='RBEP' if rbep else '1TRC', labels=labels)

    def crop(self, start: float, end: float) -> Spectrum:
        """
        Crop this Spectrum to the given time interval

        Args:
            start: start time
            end: end time

        Returns:
            the cropped Spectrum.

        """
        partials = []
        for p in self.partials:
            p2 = p.crop(start, end)
            if p2:
                partials.append(p2)
        return Spectrum(partials, indexTimeResolution=self._indexTimeResolution)

    def timeShift(self, offset: float) -> Spectrum:
        """
        Shift all partials in time by the given offset

        Args:
            offset: the time to shift the partials.

        Returns:
            the shifted Spectrum
        """
        partials = []
        for p in self.partials:
            data = p.data.copy()
            data[:, 0] += offset
            partials.append(Partial(data, label=p.label))
        return Spectrum(partials)

    def clone(self, *,
              partials: list[Partial] = None,
              indexTimeResolution: float = None):
        return Spectrum(partials or self.partials,
                        indexTimeResolution=indexTimeResolution or self._indexTimeResolution)

    def timeScaleOffsets(self, factor: float, reference=0.) -> Spectrum:
        """
        Similar to :meth:`Spectrum.timeScale` but only transforms the offsets

        The partial retains its duration and all breakoints but the offset
        are shifted relative to the first one. This is useful for applying
        a time transform to transient partials

        Args:
            factor: the scaling factor
            reference: the center of the transformation

        Returns:
            the modified Spectrum

        Example
        ~~~~~~~

        Stretch a spectrum, but only the voiced partials, to avoid smearing the
        transients

            >>> import maelzel.partialtracking as pt
            >>> import sndfileio
            >>> samples, sr = sndfileio.sndread("path/to/soundfile")
            >>> spectrum = sp.Spectrum.analyze(samples, sr=info.samplerate)
            >>> voiced, residual = spectrum.filter(maxfreq=5000, mindur=0.05, maxbandwidth=0.0001)
            >>> stretched = voiced.timeScale(8) + residual.timeScaleOffsets(8)

        """
        partials = []
        for p in self.partials:
            data = p.data.copy()
            offset = data[0, 0]
            newoffset = (offset - reference) * factor + reference
            data[:, 0] += (newoffset - offset)
            partials.append(Partial(data=data, label=p.label))
        return self.clone(partials=partials)

    def timeTransform(self, transform: Callable[[np.ndarray], np.ndarray]) -> Spectrum:
        """
        Apply a time transformation to this Spectrum

        Args:
            transform: a function maping times to times

        Returns:
            the transformed Spectrum

        Example
        ~~~~~~~

        A simple scaling factor

            >>> from maelzel.pitchtracking import Spectrum
            >>> sp = Spectrum.analyze(...)
            >>> sp2 = sp.timeTransform(lambda times: times * 2)

        A time varying scaling

            >>> import bpf4
            >>> from maelzel.pitchtracking import Spectrum
            >>> sp = Spectrum.analyze(...)
            >>> transform = bpf4.linear(0, 0, 1, 2, 2, 10, 3, 100).keep_slope()
            >>> sp2 = sp.timeTransform(transform.map)
        """
        partials = [p.timeTransform(transform) for p in self.partials]
        return Spectrum(partials)

    def timeScale(self, factor: float, reference=0.) -> Spectrum:
        """
        Scale the times of this Spectrum

        Args:
            factor: the scaling factor to apply
            reference: the center of the transformation (this point remains invariant)

        Returns:
            the scaled Spectrum
        """
        partials = []
        for p in self.partials:
            data = p.data.copy()
            times = data[:, 0]
            times *= factor
            times -= reference * (factor + 1)
            partials.append(Partial(data, label=p.label))
        return Spectrum(partials)

    def freqTransform(self, transform: Callable[[np.ndarray], np.ndarray]) -> Spectrum:
        """
        Apply a transformation to the frequencies of this spectrum

        Args:
            transform: a function mapping frequencies to frequencies

        Returns:
            the transformed spectrum


        Example
        ~~~~~~~

        Transpose a spectrum a 4th up in the pitch space

            >>> from maelzel.pitchtracking import Spectrum
            >>> from maelzel import pitchtoolsnp as ptnp
            >>> sp = Spectrum.analyze(...)
            >>> def transpose(freqs, interval):
            ...     pitches = ptnp.f2m(freqs)
            ...     return ptnp.m2f(pitches + interval)
            >>> # Transpose a partial a 4th up
            >>> sp2 = sp.freqTransform(lambda freqs: transpose(freqs, 5))

        """
        partials = [p.freqTransform(transform) for p in self]
        return self.clone(partials=partials)

    def synthesize(self, sr=44100, start=0., end=0., fadetime: float | None = None
                   ) -> audiosample.Sample:
        """
        Synthesize the partials as audio samples

        Args:
            sr: the samplerate
            start: start time of synthesis.
            end: end time of synthesis.
            fadetime: any partial starting or ending at a non-zero amplitude will
                be faded in or out using this fadetime to avoid clicks.
                Use None to use a default

        Returns:
            the generated samples as a numpy array.

        Example
        ~~~~~~~

            >>> from maelzel.partialtracking import spectrum
            >>> from sndfileio import *
            >>> samples, sr = sndread("path/to/soundfile.wav")
            >>> sp = spectrum.analyze(samples, sr=sr, resolution=50)
            >>> resynthesized = sp.synthesize()
            >>> sndwrite("resynth.wav", samples, sr=sr)

        """
        if end == 0:
            end = -1
        arrays = [p.data for p in self.partials]
        samples = lt.synthesize(arrays, samplerate=sr, start=start, end=end,
                                fadetime=fadetime if fadetime is not None else -1)
        return audiosample.Sample(samples, sr=sr)

    def play(self,
             speed=1.,
             freqscale=1.,
             gain=1.,
             bwscale=1.,
             loop=False,
             start=0.,
             stop=0.,
             minfreq=0,
             maxfreq=0,
             gaussian=False,
             interpfreq=True,
             interposcil=True,
             chan=1.,
             engine: csoundengine.Engine | str | None = None
             ) -> csoundengine.synth.Synth:
        """
        Play this spectrum in realtime

        Playback is performed via csound using the beosc opcodes.

        Args:
            speed: the playback speed (does not affect the pitch)
            loop: should playback be looped?
            start: start time of the played selection
            stop: stop time of the played selection
            minfreq: min. frequency to play
            maxfreq: max. frequency to play (0 to disable low pass filter)
            gaussian: if True, use gaussian noise for residual resynthesis
            interpfreq: if True, interpolate frequency between cycles
            interposcil: if True, use linear interpolation for the oscillators
            engine: engine used for playback. Use None for default
            chan: channep and position. An integer value indicates the first channel
                to use, a fractional value indicates channel and stereo position.
                Channels start with 1, so 1.5 indicates center position within
                the channels 1 and 2

        Returns:
            a csoundengine.Synth. It can be used to modulate / automate any dynamic parameters, such
            as speed, minfreq, etc.

        .. seealso:: :meth:`Spectrum.packMatrix`

        """
        if self._packedMatrix is None:
            self._packedMatrix = self.packMatrix()
        if engine is not None:
            if isinstance(engine, str):
                engine = _csoundEngine(engine)
        else:
            engine = self._soundEngine or _csoundEngine()
        self._soundEngine = engine
        session = engine.session()
        position = chan - int(chan)
        return session.playPartials(source=self._packedMatrix,
                                    speed=speed,
                                    loop=loop,
                                    start=start,
                                    stop=stop,
                                    minfreq=minfreq,
                                    maxfreq=maxfreq,
                                    freqscale=freqscale,
                                    gaussian=gaussian,
                                    interposcil=interposcil,
                                    interpfreq=interpfreq,
                                    chan=int(chan),
                                    gain=gain,
                                    bwscale=bwscale,
                                    position=position)

    def packMatrix(self, outfile='', maxtracks=0, period: float | None = None) -> np.ndarray:
        """
        Pack this Spectrum as a 2D matrix, used for playback

        Args:
            outfile: if given, the matrix will be saved to this file. Supported formats: .mtx, .npy.
                The .mtx format is a 32-bit float wav file with metadata containing information about
                the sampling process. The .npy format is a numpy dump format
            maxtracks:
            period: a value of None will calculate a default sampling period

        Returns:
            the packed matrix

        Example
        ~~~~~~~

        TODO

        .. seealso:: :meth:`Spectrum.play`
        """
        arrays = [p.data for p in self.partials]
        tracks, matrix = lt.util.partials_save_matrix(arrays, outfile=outfile, dt=period, maxtracks=maxtracks)
        return matrix

    def plot(self,
             axes=None,
             linewidth=1,
             avg=True,
             cmap='inferno',
             downsample=1):
        """
        Plot this spectrum using matplotlib

        Args:
            axes: if given, use this axes to plot into
            linewidth: the linewidth of the plot
            avg: if True, use the average of two breakpoints for the amp value of a line between those breakpoints
            cmap: the colormap to use
            downsample: if higher than 1, downsample the data by the given amount

        Returns:
            the axes used

        """
        from . import plotting
        plotting.plotmpl(self, axes=axes, linewidth=linewidth, avg=avg, cmap=cmap, downsample=downsample)

    def histogram(self, metric='energy', loudnessCompensation=True) -> histogram.Histogram:
        """
        Calculate a histogram over the partials of this spectrum.

        The possible metrics are 'energy', 'bandwidth' and 'duration'

        Args:
            metric: the metric to evaluate. One of 'energy', 'bandwidth' or 'duration'
            loudnessCompensation: if True and relevant for the metric, the data
                is weighted by the loudness compensation curve (ANSA A-Weighting curve)

        Returns:
            a :class:`maelzel.histogram.Histogram`

        """
        if metric == 'energy':
            if loudnessCompensation:
                data = [p.audibility() for p in self.partials]
            else:
                data = [p.energy() for p in self.partials]
        elif metric == 'bandwidth':
            data = [p.meanbw() for p in self.partials]
        else:
            raise ValueError(f"Expected one of 'energy', 'bandwidth', got {metric}")
        return histogram.Histogram(data)

    def filter(self,
               mindb: int | float = -120,
               maxfreq: int | float = 24000,
               minfreq: int | float = 0,
               mindur=0.,
               minbreakpoints=2,
               minpercentile=0.,
               loudnessCompensation=True,
               numbands=5,
               maxbandwidth=1.,
               minbandwidth=0.,
               banddistribution=0.7
               ) -> tuple[Spectrum, Spectrum]:
        """
        Filter the partials in this spectrum

        Args:
            mindb: the min. average amplitude (in dB) of a partial
            maxfreq: the max. average frequency
            minfreq: the min. average frequency
            mindur: the min. duration
            minbreakpoints: the min. number of breakpoints
            minpercentile: the min. energy percentile
            loudnessCompensation: use loudness compensation when calculating energy
            numbands: only relevant if minpercentile > 0. If greater than 1, the
                spectrum is first split into bands and the min. percentile is applied
                to each band. `banddistribution` determines the weighting of the bands
            maxbandwidth: the maximum bandwidth (noisyness) for a partial to be selected
            minbandwidth: the minimum bandwidth (noisyness) for a partial to be selected
            banddistribution: the weight distribution of the bands. A value below
                1 will compress the weight on the lower part of the spectrum, while
                a value higher than 1 will give more weight to the upper part of the
                spectrum

        Returns:
            a tuple (selected Spectrum, residual Spectrum)

        Example
        ~~~~~~~

        Select the voiced part of the spectrum

            >>> from maelzel.partialtracking import Spectrum
            >>> import sndfileio
            >>> samples, sr = sndfileio.sndread('path/to/soundfile')
            >>> spectrum = Spectrum.analyze(samples, sr=sr, resolution=50)
            >>> selected, residual = spectrum.filter(minpercentile=0.05, maxbandwidth=0.001)
        """
        if numbands > 1 and minpercentile > 0:
            bands = self.splitInBands(numbands=numbands, distribution=banddistribution)
            spectralbands: list[Spectrum] = [Spectrum(band.partials) for band in bands]
            allselected, allresidual = [], []
            for i, band in enumerate(spectralbands):
                bandminfreq = bands[i].minfreq
                bandmaxfreq = bands[i].maxfreq
                selected, residual = band.filter(mindb=mindb,
                                                 maxfreq=bandmaxfreq,
                                                 minfreq=bandminfreq,
                                                 mindur=mindur,
                                                 numbands=1,
                                                 minbreakpoints=minbreakpoints,
                                                 minpercentile=minpercentile,
                                                 loudnessCompensation=loudnessCompensation,
                                                 minbandwidth=minbandwidth,
                                                 maxbandwidth=maxbandwidth)
                allselected.extend(selected)
                allresidual.extend(residual)
            return Spectrum(allselected), Spectrum(allresidual)

        selected, residue = [], []
        minamp = pt.db2amp(mindb)
        if minpercentile > 0:
            energyhist = self.histogram(metric='energy', loudnessCompensation=loudnessCompensation)
            minenergy = energyhist.percentileToValue(minpercentile)
        else:
            minenergy = 0.

        energyfunc = (lambda p: p.audibility()) if loudnessCompensation else (lambda p: p.energy())

        for partial in self.partials:
            freq = partial.meanfreq()
            if (partial.meanamp() >= minamp and
                    minfreq <= freq <= maxfreq and
                    partial.duration >= mindur and
                    len(partial) >= minbreakpoints and
                    minbandwidth <= partial.meanbw() < maxbandwidth and
                    energyfunc(partial) >= minenergy):
                selected.append(partial)
            else:
                residue.append(partial)

        return Spectrum(selected), Spectrum(residue)

    @classmethod
    def analyze(cls,
                samples: np.ndarray | str,
                sr: int = 44100,
                resolution: float = 50.,
                windowsize: float = None,
                hoptime: float = None,
                freqdrift: float = None,
                minbreakpoints=1,
                mindb=-90,
                ) -> Spectrum:
        """
        Analyze audiosamples to generate a Spectrum via partial tracking

        The backend used is loris via loristrck

        Args:
            samples: the samples to analyze. A 1D numpy array containing the audio samples
                as floats between -1 and 1 or the path to a soundfile. If a multichannel
                array / soundfile is given, only the first channel is used
            sr: the samplerate
            resolution: the analysis resolution, determines the fft size. As a rule of thumb, somewhat lower than the
                lowest frequency expected for a monophonic source.
            windowsize: The window size in hz. This value needs to be higher than the resolution since the window in
                samples needs to be smaller than the fft analysis
            hoptime: the time to move the window after each analysis. For overlap==1, this is 1/windowsize.
                For overlap==2, 1/(windowsize*2)
            freqdrift: the max. variation in frequency between two breakpoints (by default, 1/2 resolution)
            minbreakpoints: the min. number of breakpoints a partial can have. If set to > 1, any unmatched breakpoint
                will be removed)
            mindb: the amplitude floor, in dB. Breakpoints below this amplitude will
                not be considered.

        Returns:
            the resulting Spectrum

        Example
        ~~~~~~~

            >>> import maelzel.partialtracking as pt
            >>> import sndfileio
            >>> samples, sr = sndfileio.sndread("path/to/sndfile")
            >>> spectrum = pt.analyze
        """
        if isinstance(samples, str):
            import sndfileio
            samples, sr = sndfileio.sndread(samples)

        if len(samples.shape) == 2:
            samples = samples[:, 0]
        partialarrays = lt.analyze(samples,
                                   sr=sr,
                                   resolution=resolution,
                                   windowsize=windowsize or -1,
                                   hoptime=hoptime or -1,
                                   ampfloor=mindb,
                                   freqdrift=freqdrift or -1)
        if minbreakpoints > 1:
            partialarrays = [p for p in partialarrays if len(p) >= minbreakpoints]

        return Spectrum(partials=[Partial(data) for data in partialarrays])

    def fundamental(self, minfreq=60., method='pyin') -> tuple[bpf4.BpfInterface, bpf4.BpfInterface]:
        """
        Extract the fundamental of this spectrum

        Args:
            minfreq: the min. frequency of the fundamental.
            method: the method used. At the moment only 'pyin' is supported.

        Returns:
            a tuple (f0curve: BpfInterface, voicedness: BpfInterface), where `f0curve` is a bpf
            mapping time to frequency and `voicedness` is a bpf mapping time to prediction confidence.

        """
        sr = 44100
        sample = self.synthesize(sr=sr)
        from maelzel.snd import freqestimate
        f0curve, voicedness = freqestimate.f0curve(sample.samples, sr=sample.sr)
        return f0curve, voicedness

    def splitInBands(self, numbands: int, distribution: float | bpf4.core.BpfInterface = 1.0
                     ) -> list[pack.SpectralBand]:
        """
        Split this spectrum into bands

        The actual frequencies of each band depends on the energy distribution
        of the spectrum itself

        Args:
            numbands: the number of bands
            distribution: the energy distribution. If < 1, more resolution is given to the
                lower frequencies. A distribution > 1 will result in bands with higher
                resolution for the higher frequencies. A bpf bapping (0, 0) to (1, 1) is
                also possible

        Returns:
            a list of :class:`maelzel.partialtracking.pack.SpectralBand`, where a
            :class:`SpectralBand` is a dataclass with attributes `minfreq` (min.
            frequency of the band); `maxfreq` (the max. frequency of the band); and
            `partials` (the partials in this band).

        """
        return pack.splitInBands(partials=self.partials, numbands=numbands,
                                 distribution=distribution)

    def splitInTracks(self,
                      maxtracks: int,
                      noisetracks: int = 0,
                      maxrange: int = 36,
                      relerror=0.05,
                      distribution: float | Callable[[float], float] | list[float] | None = 2,
                      numbands: int = None,
                      mingap=0.1,
                      audibilityCurveWeight=1.,
                      noisebw=0.001,
                      noisefreq=3500,
                      minbreakpoints=1
                      ) -> pack.SplitResult:
        """
        Split this spectrum into tracks

        A track is a list of non-overlapping partials. The partials within a track might be played
        back by a single oscillator or displayed in a single line.

        This is used, for example, to render a Spectrum into notation.

        Args:
            maxtracks: the max. number of regular tracks
            noisetracks: number of tracks used for noise
            maxrange: the max. range for regular tracks
            relerror: the relative error when splitting in tracks. A larger value will result in a number of
                tracks which might be higher than the actual maxtracks value
            distribution: an exponential, a function mapping to distribute partial weights across a number of bands,
                a list of exponentials or None. A value between 0 and 1 will add more resolution to the lower frequencies,
                a value higher than 1 will shift resolution to the higher frequencies.
                If None, distribution is optimized to maximize audibility (this will call the packing routine multiple times
                and might result in a long computation). A list of values will use those values to find the best distribution.
                In all cases the variable maximized is the total audibility (audibility being the energy contributed by the
                partial weighted by the ANSI A-Weighting loudness curve)
            numbands: before splitting into tracks the spectrum is divided into bands following the distribution. If the
                distribution is 1, all bands will have the same energy. The splitting frequencies of the bands depend on
                the actual energy distribution within the spectrum
            mingap: the min. silence between two partials within a track
            noisebw:
            audibilityCurveWeight: the weight of the frequency dependent amplitude curve when determining the importance
                of a partial. A value of 1 will give full weight to this curve, a value of 0 will only use the amplitude
                and duration of the partial to calculate its energy. Values in between are possible.

        Returns:
            a :class:`~maelzel.partialtracking.pack.SplitResult`, which is a dataclass with attributes:

            * 1) `tracks`: the fitted tracks
            * 2) `noisetracks`: if called with `noisetracks` > 0, partials considered noise will be fitted within
                these tracks
            * 3) `residual`: a list of Partials not included in `tracks` and `noisetracks`
            * 4) `distribution`: the frequency distribution used. This is only relevant if asked to optimize this parameter


        Example
        ~~~~~~~

            >>> from maelzel.partialtracking.spectrum import Spectrum
            >>> import sndfileio
            >>> samples, sr = sndfileio.sndread('path/to/soundfile')
            >>> spectrum = Spectrum.analyze(samples, sr=sr, resolution=50)
            >>> # Split into 10 tracks and distribute noise across 4 extra tracks
            >>> splitresult = spectrum.splitInTracks(maxtracks=10, noisetracks=4)
            >>> # TODO: do something interesting with the result

        """
        if minbreakpoints > 1:
            partials = [p for p in self.partials if p.numbreakpoints >= minbreakpoints]
        else:
            partials = self.partials

        if isinstance(distribution, (int, float, bpf4.BpfInterface)):
            tracks, residualtracks, unfittedpartials = pack.splitInTracks(
                partials,
                maxtracks=maxtracks,
                maxrange=maxrange,
                relerror=relerror,
                distribution=distribution,
                numbands=numbands,
                mingap=mingap,
                audibilityCurveWeight=audibilityCurveWeight,
                maxnoisetracks=noisetracks,
                noisefreq=noisefreq,
                noisebw=noisebw)
            return pack.SplitResult(distribution=distribution,
                                    tracks=tracks,
                                    noisetracks=residualtracks,
                                    residual=unfittedpartials)
        else:
            return pack.optimizeSplit(self.partials,
                                      maxtracks=maxtracks,
                                      maxrange=maxrange,
                                      relerror=relerror,
                                      distributions=distribution,
                                      numbands=numbands,
                                      mingap=mingap,
                                      noisetracks=noisetracks,
                                      noisefreq=noisefreq,
                                      noisebw=noisebw)


analyze = Spectrum.analyze
