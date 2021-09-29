"""
utility functions to work with music21 which are specific to maelzel.core
(they might depend on the current config) and should not be moved
to maelzel.music.m21tools
"""
import music21 as m21
from ._common import *
from maelzel.music import m21tools
from emlib import iterlib

from . import tools
from .workspace import activeConfig


def m21Note(pitch: U[str, float], showcents:bool=None, divsPerSemitone:int=None,
            config=None, **options) -> m21.note.Note:
    """
    Create a m21.note.Note, taking semitoneDivisions into account

    Args:
        pitch: a notename or a midinote, possibly with fractional part
        showcents: if True, attach the cents representation as a lyric
        divsPerSemitone: divisions of the semitone
        config: the config to use, or None to use default
        options: options passed to m21.note.Note

    Returns:
        the generated note
    """
    assert isinstance(pitch, (str, int, float))
    if config is None:
        config = activeConfig()
    divs = divsPerSemitone or config['semitoneDivisions']
    showcents = showcents if showcents is not None else config['show.cents']
    note, centsdev = m21tools.makeNote(pitch, divsPerSemitone=divs,
                                       showcents=showcents, **options)
    return note


def m21MicrotonalNote(pitch: float, duration, showcents:bool=None, divsPerSemitone:int=None, 
                      config=None, **options):
    config = config or activeConfig()
    divs = divsPerSemitone or config['semitoneDivisions']
    basepitch = tools.quantizeMidi(pitch, 1/divs)
    note = m21Note(basepitch, quarterLength=duration, divsPerSemitone=divs,
                   showcents=showcents, **options)
    cents = int((pitch - basepitch) * 100)
    note.microtone = cents
    return note


def m21Chord(midinotes:Seq[float], showcents=None, 
             config=None, **options
             ) -> m21.chord.Chord:
    """
    Create a m21 Chord out of a seq. of midinotes
    """
    # m21chord = m21.chord.Chord([m21.note.Note(n.midi) for n in notes])
    assert all(isinstance(note, (str, int, float)) for note in midinotes)
    config = config or activeConfig()
    divsPerSemi = config['semitoneDivisions']
    if showcents is None:
        showcents = config['show.cents']
    chord, cents = m21tools.makeChord(midinotes, showcents=showcents, divsPerSemitone=divsPerSemi,
                                      **options)
    return chord


def m21TextExpression(text:str, style:str=None, config=None) -> m21.expressions.TextExpression:
    """
    style: one of None (default), 'small', 'bold', 'label'
    """
    txtexp = m21.expressions.TextExpression(text)
    config = config or activeConfig()
    if style == 'small':
        txtexp.style.fontSize = 12.0
        txtexp.style.letterSpacing = 0.5
    elif style == 'bold':
        txtexp.style.fontWeight = 'bold'
    elif style == 'label':
        txtexp.style.fontSize = config['show.labelFontSize']
    return txtexp


def m21Label(text:str, config=None) -> m21.expressions.TextExpression:
    return m21TextExpression(text, style='label', config=config)


def bestClef(midinotes: Iter[float]) -> m21.clef.Clef:
    """
    Return a m21 Clef which best fits the given pitches

    Args:
        midinotes: the pitches to

    Returns:

    """
    mean = iterlib.avg(midinotes, None)
    if mean is None:
        # no notes
        return m21.clef.TrebleClef()
    elif mean > 90:
        return m21.clef.Treble8vaClef()
    elif mean > 59:
        return m21.clef.TrebleClef()
    else:
        return m21.clef.BassClef()
