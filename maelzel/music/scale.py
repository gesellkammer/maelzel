"""
Create and work with musical scales

"""
from __future__ import annotations
from pitchtools import n2m
from emlib import iterlib
from typing import Union, Sequence


knownscales = {
    'major': (2, 2, 1, 2, 2, 2, 1),
    'minor-harmonic': (2, 1, 2, 2, 2, 2, 1),
    'octotonic1': (2, 1, 2, 1, 2, 1, 2, 1),
    'octotonic2': (1, 2, 1, 2, 1, 2, 1, 2),
    'chromatic': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
}


def _asmidi(x: Union[str, float]) -> float:
    return n2m(x) if isinstance(x, str) else x


def pitchscale(startpitch: Union[float, str],
               steps: Union[str, Sequence[float]] = 'chromatic',
               endpitch: Union[float, str] = "8C"
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
        steps = knownscales.get(steps)
        if steps is None:
            raise ValueError(f"steps should be either a sequence of intervals or the name of"
                             f"a known interval sequence. Known sequences: {knownscales.keys()}")
    midinotes = [startpitch]
    for step in iterlib.cycle(steps):
        if (midinote := midinotes[-1] + step) > endpitch:
            break
        midinotes.append(midinote)
    return midinotes
