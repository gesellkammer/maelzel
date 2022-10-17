"""
functions to fix errors in music21
"""
from __future__ import annotations
import music21 as m21
from music21.common.numberTools import opFrac
import logging
import copy
from typing import List


logger = logging.getLogger(__name__)


def makeTupletBrackets(s: m21.stream.Stream, inPlace=False) -> m21.stream.Stream:
    returnObj = s if inPlace else copy.deepcopy(s)
    durationList = [n.duration for n in returnObj.notesAndRests if n.duration.quarterLength > 0]
    if not durationList:
        logger.info(f"No notes or rests to make subdivision brackets on stream: {s}")
        return returnObj
    tupletMap = []  # a list of (subdivision obj / Duration) pairs
    for dur in durationList:  # all Duration objects
        tupletList = dur.tuplets
        if tupletList in [(), None]:  # no tuplets, length is zero
            tupletMap.append([None, dur])
        elif len(tupletList) > 1:
            logger.warning('got multi-subdivision duration; cannot yet handle this. %s' % repr(tupletList))
        elif len(tupletList) == 1:
            tupletMap.append([tupletList[0], dur])
            if tupletList[0] != dur.tuplets[0]:
                raise Exception('cannot access Tuplets object from within DurationTuple.')
        else:
            raise Exception('cannot handle these tuplets: %s'%tupletList)

    # have a list of subdivision, Duration pairs
    completionCount = 0  # qLen currently filled
    completionTarget = None  # qLen necessary to fill subdivision
    for i in range(len(tupletMap)):
        tupletObj, dur = tupletMap[i]
        tupletPrevious = tupletMap[i-1][0] if i>0 else None
        tupletNext = tupletMap[i+1][0] if i<len(tupletMap)-1 else None
        if tupletObj is not None:
            completionCount = opFrac(completionCount+dur.quarterLength)
            # if previous subdivision is None, always start. Always reset completion target
            if tupletPrevious is None or completionTarget is None:
                if tupletNext is None:  # single subdivision w/o tuplets either side
                    tupletObj.type = 'startStop'
                    tupletObj.bracket = False
                    completionCount = 0  # reset
                else:
                    tupletObj.type = 'start'
                    completionTarget = tupletObj.totalTupletLength()
                    # if subdivision next is None, always stop
            # if both previous and next are None, just keep a start

            # this, below, is optional:
            # if next normal type is not the same as this one, also stop
            elif tupletNext is None or completionCount >= completionTarget:
                tupletObj.type = 'stop'  # should be impossible once frozen...
                completionTarget = None  # reset
                completionCount = 0  # reset
            elif tupletPrevious is not None and tupletNext is not None:
                # do not need to change subdivision type; should be None
                pass
    return returnObj


def makeBeams(s: m21.stream.Stream, *, inPlace=False) -> m21.stream.Stream:
    returnObj = s if inPlace else copy.deepcopy(s)
    mColl: List[m21.stream.Measure]
    if 'Measure' in s.classes:
        mColl = [returnObj]  # store a list of measures for processing
    else:
        mColl = list(returnObj.iter.getElementsByClass('Measure'))  # a list of measures
        if not mColl:
            raise m21.stream.StreamException('Cannot process a stream that is neither a Measure nor has no Measures')

    lastTimeSignature = None
    for m in mColl:
        # this means that the first of a stream of time signatures will be used
        if m.timeSignature is not None:
            lastTimeSignature = m.timeSignature
        if lastTimeSignature is None:
            raise m21.stream.StreamException('cannot process beams in a Measure without a time signature')
        noteGroups = []
        if m.hasVoices():
            for v in m.voices:
                noteGroups.append(v.notesAndRests.stream())
        else:
            noteGroups.append(m.notesAndRests.stream())

        for noteStream in noteGroups:
            if len(noteStream) <= 1:
                continue  # nothing to beam
            realNotes = [n for n in noteStream if n.durationSecs.quarterLength > 0]
            durList = [n.durationSecs for n in realNotes]
            # error check; call before sending to time signature, as, if this
            # fails, it represents a problem that happens before time signature
            # processing
            summed = sum([d.quarterLength for d in durList])
            durSum = opFrac(opFrac(summed))  # the double call corrects for tiny errors in adding

            # floats and Fractions in the sum() call -- the first opFrac makes it
            # impossible to have 4.00000000001, but returns Fraction(4, 1). The
            # second call converts Fraction(4, 1) to 4.0
            barQuarterLength = lastTimeSignature.barDuration.quarterLength
            if durSum > barQuarterLength:
                continue

            # getBeams
            offset = 0.0
            if m.paddingLeft != 0.0:
                offset = opFrac(m.paddingLeft)
            elif noteStream.highestTime < barQuarterLength:
                offset = barQuarterLength-noteStream.highestTime

            beamsList = lastTimeSignature.getBeams(realNotes, measureStartOffset=offset)
            for n, beams in zip(realNotes, beamsList):
                n.beams = beams if beams is not None else m21.beam.Beams()

    del mColl  # remove Stream no longer needed

    returnObj.streamStatus.beams = True
    return returnObj


def fixStream(s: m21.stream.Stream, inPlace=False) -> m21.stream.Stream:
    """
    Call this prior to calling .show()
    """
    out = s if inPlace else copy.deepcopy(s)
    if isinstance(out, m21.stream.Score):
        for part in out.parts:
            fixStream(part, inPlace=True)
        return out
    out.makeMeasures(inPlace=True)
    for meas in out.iter.getElementsByClass('Measure'):
        makeTupletBrackets(meas, inPlace=True)
        try:
            makeBeams(meas, inPlace=True)
        except:
            pass
    return out


def show(s: m21.stream.Stream, *args, **kws):
    fixed = fixStream(s, inPlace=False)
    fixed.show(*args, **kws)