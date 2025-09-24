"""
Create and work with musical scales

"""
from __future__ import annotations
import pitchtools as pt
from emlib import iterlib
from typing import Sequence
from functools import cache


knownscales = {
    'major': (2, 2, 1, 2, 2, 2, 1),
    'aeolic': (2, 1, 2, 2, 2, 1, 2),
    'minor-harmonic': (2, 1, 2, 2, 2, 2, 1),
    'octotonic1': (2, 1, 2, 1, 2, 1, 2, 1),
    'octotonic2': (1, 2, 1, 2, 1, 2, 1, 2),
    'chromatic': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
}


def _asmidi(x: str | float) -> float:
    return pt.n2m(x) if isinstance(x, str) else x


@cache
def _scale(startnote: str,
           steps: tuple[float, ...],
           endnote='8C'
           ) -> list[str]:
    endpitch = pt.n2m(endnote)
    note = startnote
    pitch = pt.n2m(note)
    notes = [note]
    for step in iterlib.cycle(steps):
        pitch += step
        if pitch > endpitch:
            break
        note = pt.transpose(note, step)
        notes.append(note)
    return notes


@cache
def _pitchscale(startpitch: float,
                steps: tuple[float, ...],
                endpitch: float
                ) -> list[float]:
    midinotes = [startpitch]
    for step in iterlib.cycle(steps):
        if (midinote := midinotes[-1] + step) > endpitch:
            break
        midinotes.append(midinote)
    return midinotes


def pitchscale(startpitch: float | str,
               steps: str | Sequence[float] = 'chromatic',
               endpitch: float | str = "8C"
               ) -> list[float]:
    """
    Create a pitch scale

    Args:
        startpitch: the starting pitch of the scale
        steps: a sequence of semitones. This sequence is cycled until the
            *endpitch* is reached. Values can be fractional and they do not
            necessarily need to add up to 12. Instead of the steps a preset string
            can be given (see *knownscales*)
        endpitch: the maximum pitch of the scale.

    Returns:
        a list of midinotes

    Example
    ~~~~~~~

        >>> from maelzel.music.scale import *
        >>> from pitchtools import *
        >>> cmajor = [m2n(m) for m in pitchscale('3C', steps='major', endpitch='5C')]
        >>> ' '.join(cmajor)
        3C 3D 3E 3F 3G 3A 3B 4C 4D 4E 4F 4G 4A 4B 5C
    """
    startpitch = _asmidi(startpitch)
    endpitch = _asmidi(endpitch)

    if isinstance(steps, str):
        scale = knownscales.get(steps)
        if scale is None:
            raise ValueError(f"steps should be either a sequence of intervals or the name of"
                             f"a known interval sequence. Known sequences: {knownscales.keys()}")
    elif isinstance(steps, tuple):
        scale = steps
    else:
        scale = tuple(steps)

    return _pitchscale(startpitch=startpitch, steps=scale, endpitch=endpitch)
