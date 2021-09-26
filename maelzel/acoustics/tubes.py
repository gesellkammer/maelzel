"""
Pipe acoustics
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *


def tube_freq(length:float, width:float, kind='open', shape="circular", c=343.0) -> float:
    """
    Calculate the fundamental of an organ tube

    Args:
        length: length of the tube
        kind: 'open' or 'closed'
        c: sound-speed
        width: diameter of the tube, used to calculate a correction factor
                Use 0 to skip, or the side for a square shape
        shape: "circular" or "square"

    Returns:
        the resonating frequency
    """
    if kind == 'open':
        if shape == "circular":
            correction = 0.6 * width
        else:
            correction = -(2 * width)
        L = length + correction
        lmbda = 2 * L
        f = c / lmbda
    elif kind == 'closed':
        if shape == "circular":
            correction = 0.3 * width
        else:
            correction = -width
        L = length + correction
        lmbda = 4 * L
        f = c / lmbda
    else:
        raise ValueError(f"kind {kind} unknown, possible values: 'open', 'closed'")
    return f


def organ_pipe_length(freq:float, width_ratio=12.0, kind='closed', footmount=0.02, 
                      c=343.0) -> float:
    """
    Calculate the pipe length for the given conditions

    Args:
        freq: the desired sounding frequency
        width_ratio: the width to length ratio
        kind: 'closed' or 'open'
        footmount: the height of the foot mount.
        c: speed of sound

    Returns:
        the organ pipe length
    """
    lmbda = c / freq
    L = lmbda / 2
    W = L / width_ratio
    F = footmount if footmount >= 0 else W
    if kind == 'closed':
        Lpipe = 0.52 * L + W + F
    elif kind == 'open':
        Lpipe = 1.03 * L - W + F
    else:
        raise ValueError("The pipe must be open or closed")
    return Lpipe


def organ_slide_length(freq:float, width_ratio=12.0, c=343.0) -> float:
    return organ_pipe_length(freq=freq, width_ratio=width_ratio, kind='closed',
                             footmount=2, c=c)

