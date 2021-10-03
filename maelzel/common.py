from typing import Tuple, Union as U, TypeVar
from maelzel.rational import Rat as F
from pitchtools import n2m


pitch_t = U[int, float, str]
timesig_t = Tuple[int, int]
number_t = U[int, float, F]
T = TypeVar('T')


def asmidi(x: pitch_t) -> float:
    """
    Converts a notename to a midinote.
    """
    if isinstance(x, (int, float)):
        if x > 127:
            raise ValueError("A midinote expected (< 128), but got a value of {x}!")
        return x
    elif isinstance(x, str):
        return n2m(x)
    try:
        return float(x)
    except TypeError:
        raise TypeError(f"could not convert {x} to a midi note")

