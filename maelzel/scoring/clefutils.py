from __future__ import annotations
from functools import cache
import bpf4
import visvalingamwyatt
from emlib import iterlib
from pitchtools import n2m
from .notation import Notation
from .common import logger
from . import attachment
from . import definitions

from typing import Sequence


@cache
def clefEvaluators() -> dict[str, bpf4.BpfInterface]:
    return {
        'treble': bpf4.linear(
            (n2m("1C"), -20),
            (n2m("2A"), -2),
            (n2m("3F"), 0),
            (n2m("3B"), 1),
            (n2m("6E"), 1),
            (n2m("6B"), 0),
            (n2m("7A"), -2),
        ),
        'treble8': bpf4.linear(
            (n2m("4G"), -2),
            (n2m("5C"), 0),
            (n2m("5F"), 1),
            (n2m("7C"), 1),
            (n2m("7E"), 0)),
        'treble15': bpf4.linear(
            (n2m("5C"), -2),
            (n2m("6C"), 0),
            (n2m("6F"), 1),
            (n2m("10C"), 1)),
        'bass': bpf4.linear(
            (n2m("1A"), 0),
            (n2m("2C"), 1),
            (n2m("4D"), 1),
            (n2m("4G"), 0),
            (n2m("5D"), -2)
        ),
        'bass8': bpf4.linear(
            (n2m("0C"), 1),
            (n2m("3D"), 1),
            (n2m("3A"), 0),
            (n2m("4E"), -2)
        ),
    }


def bestclef(notations: Sequence[Notation], biasclef='', biasfactor=1.5) -> str:
    pointsPerClef = {clef: sum(evaluator(p) for n in notations if not n.isRest
                               for p in n.pitches)
                     for clef, evaluator in clefEvaluators().items()}
    if biasclef:
        pointsPerClef[biasclef] *= biasfactor
    return max(pointsPerClef.items(), key=lambda pair: pair[1])[0]


def findBestClefs(notations: list[Notation], firstclef='', winsize=1, threshold=0.,
                  biasfactor=1.5, addclefs=False, key='clef'
                  ) -> list[tuple[int, str]]:
    """
    Given a list of notations, find the clef changes

    Args:
        notations: the notations.
        firstclef: the clef to start with
        winsize: the size of the sliding window. The bigger the window, the
            broader the context considered.
        threshold: if given, this is used to perform a simplification using
            the visvalingam-wyatt algorithm. The bigger this value, the
            sparser the clef changes. A value of 0. disables simpplification
        biasfactor: The higher this value, the more weight is given to the
            previous clef, thus making it more difficult to change clef
            more minor jumps
        addclefs: if True, add a Clef change (an attachment.Clef) to the
            notation where a clef change should happen.
        key: the property key to add to the notation to mark
             a clef change. Setting this property alone will not
             result in a clef change in the notation (see `addclefs`)


    Returns:
        a list of tuples (notationsindex, clef)
    """
    points = []
    clefbyindex = ['treble15', 'treble8', 'treble', 'bass', 'bass8']
    currentclef = firstclef
    notations = [n for n in notations if not n.isRest]
    if not notations:
        logger.debug("No pitched notations given")
        return []

    for i, group in enumerate(iterlib.window(notations, size=winsize)):
        currentclef = bestclef(list(group), biasclef=currentclef, biasfactor=biasfactor)
        points.append((i, clefbyindex.index(currentclef)))

    if threshold > 0 and len(points) > 2:
        simplifier = visvalingamwyatt.Simplifier(points)
        points = simplifier.simplify(threshold=threshold)

    out = []
    lastidx = -1
    for point in points:
        clefidx = point[1]
        if clefidx != lastidx:
            out.append(point)
            lastidx = clefidx
    clefs = [(idx, clefbyindex[clefindex]) for idx, clefindex in out]
    for idx, clef in clefs:
        n = notations[idx]
        n.setProperty(key, clef)
        if addclefs:
            n.addAttachment(attachment.Clef(clef))
    return clefs


def bestClefForPitch(pitch: float,
                     clefs: Sequence[str],
                     evaluators: dict[str, bpf4.BpfInterface] = None
                     ) -> tuple[str, float]:
    """
    Determines the most appropriate clef for the given pitch

    Args:
        pitch: the pitch
        clefs: the clefs to considere
        evaluators: the evaluators as returned via :func:`clefEvaluators`

    Returns:
        a tuple (clef: str, fitness: float), where clef is the name of the
        clef ('treble', 'bass', 'treble8', etc) and fitness indicates how good
        this clef is for representing this pitch (the higher, the more adequate
        the clef is)

    """
    if evaluators is None:
        evaluators = clefEvaluators()
    data = [(evaluators[clef](pitch), clef) for clef in clefs]
    best = max(data)
    return best[1], best[0]

def explodeNotations(notations: list[Notation],
                     maxstaves=3,
                     ) -> list[tuple[str, list[Notation]]]:
    if maxstaves == 1:
        clef = bestclef(notations)
        return [(clef, notations)]
    elif maxstaves == 2:
        possibleClefs = [
            ('treble15', 'treble'),
            ('treble15', 'bass'),
            ('treble8', 'bass'),
            ('treble8', 'bass8'),
            ('treble', 'bass'),
            ('treble', 'bass8'),
        ]
    elif maxstaves == 3:
        possibleClefs = [
            ('treble15', 'treble', 'bass'),
            ('treble15', 'treble', 'bass8'),
            ('treble', 'bass', 'bass8')
        ]
    elif maxstaves == 4:
        possibleClefs = [
            ('treble15', 'treble', 'bass', 'bass8')
        ]
    else:
        raise ValueError(f"The max. number of staves must be between between 1 and 4, "
                         f"got {maxstaves}")

    clefevals = clefEvaluators()

    results = {}
    for clefs in possibleClefs:
        totalFitness = 0
        for n in notations:
            for pitch in n.pitches:
                clef, fitness = bestClefForPitch(pitch, clefs, evaluators=clefevals)
                totalFitness += fitness
        results[clefs] = totalFitness

    bestClefCombination = max(results.items(), key=lambda pair: pair[1])[0]
    return splitNotationsByClef(notations, clefs=bestClefCombination)


def splitNotationsByClef(notations: list[Notation],
                         clefs: Sequence[str]
                         ) -> list[tuple[str, list[Notation]]]:
    if any(clef not in definitions.clefs for clef in clefs):
        clef = next(clef for clef in clefs if clef not in definitions.clefs)
        raise ValueError(f"Clef {clef} not known. Expected {definitions.clefs.keys()}")

    evaluators = clefEvaluators()
    # Normalize clefs
    clefs = [definitions.clefs[clef] for clef in clefs]
    clefs = sorted(clefs, key=lambda clef: definitions.clefSortOrder[clef])
    parts = {clef: [] for clef in clefs}
    lastn = len(notations) - 1
    for nidx, n in enumerate(notations):
        assert isinstance(n, Notation)
        if n.isRest:
            for part in parts.values():
                part.append(n.copy())
        else:
            pitchindexToClef = [n.getClefHint(i) or bestClefForPitch(p, clefs=clefs, evaluators=evaluators)[0]
                                for i, p in enumerate(n.pitches)]
            clef0 = pitchindexToClef[0]
            if len(n.pitches) == 1 or all(clef == clef0 for clef in pitchindexToClef):
                parts[clef0].append(n)
                for otherclef in clefs:
                    if otherclef != clef0:
                        parts[otherclef].append(n.asRest())
            else:
                # A chord, distribute notes within parts
                for clef in clefs:
                    indexes = [i for i, clef2 in enumerate(pitchindexToClef) if clef == clef2]
                    if not indexes:
                        if n.duration > 0:
                            parts[clef].append(n.asRest())
                        else:
                            assert n.isGracenote
                    else:
                        parts[clef].append(n.extractPartialNotation(indexes))
                        if n.gliss and nidx < lastn:
                            for idx in indexes:
                                notations[nidx+1].setClefHint(clef, idx)

    return [(clef, part) for clef, part in parts.items()
            if any(not item.isRest for item in part)]







