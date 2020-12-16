from .audiosample import Sample
import numpy as np
import os
import logging

############################
#       Timeline
############################

logger = logging.getLogger("maelzel.snd.timeline")


class Timeline(object):
    def __init__(self, samplerate):
        # type: (Seq[Sample], Seq[float]) -> None
        """
        samples: a seq. of Samples
        times: a seq. of time offsets (in seconds)
        """
        self.samplerate = samplerate
        self.timeline = []  # type: List[Tup[float, Sample]]

    @property
    def channels(self):
        # type: () -> int
        return max(sample.channels for time, sample in self.timeline)

    def add(self, time, sample):
        # type: (float, Sample) -> None
        """
        Add a new Sample at given time

        time: the time at which to add the sample
        sample: a Sample
        """
        assert isinstance(sample, Sample)
        if sample.samplerate != self.samplerate:
            logger.info("Adding a sample with a different samplerate: resampling")
            sample = sample.resample(self.samplerate)
        self.timeline.append((time, sample))
        self.timeline.sort()

    def __iter__(self):
        # type: () -> Iterator[Tup[float, Sample]]
        return iter(self.timeline)

    def write(self, outfile, samplerate):
        # type: (str, int) -> Sample
        """
        write the timeline to disk

        Returns
        =======

        the audio samples
        """
        # min_time = min(s.time for s in self.timeline)
        maxtime = max(time + sample.duration for time, sample in self)  # type: float
        numsamples = int(maxtime * self.samplerate) + 1
        out = np.zeros((numsamples,), dtype=float)
        for t, sample in self.timeline:
            if sample.samplerate != samplerate:
                sample = sample.resample(self.samplerate)
            t0_frames = int(t * sample.samplerate)
            dur_frames = sample.duration * sample.samplerate
            out[t0_frames:t0_frames + dur_frames] += sample.samples
        ext =os.path.splitext(outfile)[1][1:].lower()
        if ext == 'flac':
            bits = 24
        elif ext in ('wav', 'aif', 'aifc', 'wv'):
            bits = 32
        else:
            raise ValueError("File format not understood."
                             "The format is inferred from the extension")
        s = Sample(out, samplerate).write(outfile, bits=bits)
        return s
