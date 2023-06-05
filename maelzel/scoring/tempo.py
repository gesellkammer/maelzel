from __future__ import annotations
from dataclasses import dataclass
from .common import F


numericUnitToName = {
    1: 'whole',
    2: 'half',
    4: 'quarter',
    8: 'eighth',
    16: '16th',
    32: '32nd',
    64: '64th'
}


@dataclass
class Metronome:
    unit: int
    """The unit of reference (4=quarter, 8=eighth, ...)"""

    dots: int
    """Number of dots"""

    bpm: float
    """The metronome value"""

    @property
    def unitstr(self) -> str:
        """The unit as a string ('quarter' instead of 4, etc.)"""
        return numericUnitToName[self.unit]


def inferMetronomeMark(quarterTempo: F,
                       timesig: tuple[int, int],
                       mintempo=48.,
                       maxtempo=144.
                       ) -> Metronome:
    """
    Infer a metronome mark for a given quartertempo

    Args:
        quarterTempo: the tempo corresponding to a quarternote
        timesig: the time signature
        mintempo: the min. tempo to allow
        maxtempo: the max. tempo to allow

    Returns:
        a :class:`Metronome` object
    """
    num, den = timesig
    if den == 4:
        if quarterTempo < mintempo:
            return Metronome(unit=8, dots=0, bpm=float(quarterTempo * 2))
        elif quarterTempo < maxtempo:
            return Metronome(unit=4, dots=0, bpm=float(quarterTempo))
        else:
            if num in (2, 4):
                return Metronome(unit=2, dots=0, bpm=float(quarterTempo / 2))
            elif num in (3,):
                return Metronome(unit=2, dots=1, bpm=float(quarterTempo / 3))
            else:
                return Metronome(unit=4, dots=0, bpm=float(quarterTempo))
    elif den in (8, 16, 32, 64):
        metro = inferMetronomeMark(quarterTempo*2, timesig=(num, den//2),
                                   mintempo=mintempo, maxtempo=maxtempo)
        return Metronome(unit=metro.unit*2, dots=metro.dots, bpm=metro.bpm)
    elif den in (1, 2):
        metro = inferMetronomeMark(quarterTempo / 2, timesig=(num, den*2),
                                   mintempo=mintempo, maxtempo=maxtempo)
        return Metronome(unit=metro.unit // 2, dots=metro.dots, bpm=metro.bpm)
    else:
        raise ValueError(f"Timesignature {timesig} not supported")