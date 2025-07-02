from __future__ import annotations

from functools import cache
from emlib import iterlib
from pitchtools import n2m
from .common import logger
from . import attachment
from . import definitions

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .notation import Notation
    from typing import Sequence
    import bpf4


@cache
def clefEvaluators() -> dict[str, bpf4.BpfInterface]:
    import bpf4
    return {
        'treble15': bpf4.linear(
            (n2m("5C"), -2),
            (n2m("6C"), 0),
            (n2m("6F"), 1),
            (n2m("10C"), 1)),
        'treble8': bpf4.linear(
            (n2m("4G"), -2),
            (n2m("5C"), 0),
            (n2m("5F"), 1),
            (n2m("7C"), 1),
            (n2m("7E"), 0)),
        'treble': bpf4.linear(
            (n2m("1C"), -20),
            (n2m("2A"), -2),
            (n2m("3F"), 0),
            (n2m("3B"), 1),
            (n2m("6E"), 1),
            (n2m("6B"), 0),
            (n2m("7A"), -2),
        ),
        'bass': bpf4.linear(
            (n2m("1A"), 0),
            (n2m("2C"), 1),
            (n2m("4D"), 1),
            (n2m("4G"), 0),
            (n2m("5D"), -2)
        ),
        'bass8': bpf4.linear(
            (n2m("-1A"), -2),
            (n2m("0C"), 0),
            (n2m("0E"), 1),
            (n2m("3D"), 1),
            (n2m("3A"), 0),
            (n2m("4E"), -2)
        ),
        'bass15': bpf4.linear(
            (n2m("-1D"), 1),
            (n2m("1C"), 1),
            (n2m("1G"), 0),
            (n2m("2C"), -2)
        ),

    }


class ClefEvaluator:
    """
    A class to evaluate clef changes within a voice

    Args:
        biasFactor: how much should we bias the current clef. A higher value
            makes it more difficult to change clef
        clefChangeBetweenTiedNotes: if True, allow clef changes between tied notes
        changeDistanceFactor: a dict mapping distance to the last clef change to
            a factor affecting the likelyhood of the clef change. A value below
            1. makes it more difficult to change clef at a given distance. This is
            used in order to prevent very frequent clef changes
    """
    clefs = ['treble15', 'treble8', 'treble', 'bass', 'bass8', 'bass15']

    clefChangeFactor = {
        ('treble', 'treble8'): 0.5,
        ('treble8', 'treble15'): 0.2,
        ('treble', 'treble15'): 0.9,
        ('bass', 'treble15'): 0.9,
        ('bass', 'bass8'): 0.5,
        ('bass', 'bass15'): 0.9,
        ('bass8', 'bass15'): 0.2
    }

    defaultDistanceFactor = {
        3: 1,
        2: 0.9,
        1: 0.2,
    }

    def __init__(self,
                 biasFactor: float = 1.5,
                 clefChangeBetweenTiedNotes: bool = False,
                 changeDistanceFactor: dict[int, float] | None = None,
                 firstClef=''):
        self.biasFactor = biasFactor
        self.clefChangeBetweenTiedNotes = clefChangeBetweenTiedNotes
        self.clefEvaluators = clefEvaluators()
        self.history: list[tuple[int, str]] = []
        self.currentClef = firstClef
        self.currentIndex = 0
        self.changeDistanceFactor = changeDistanceFactor or ClefEvaluator.defaultDistanceFactor

    def _bestClef(self, notations: Sequence[Notation]) -> str:
        pointsPerClef: dict[str, float] = {}
        if all(n.isRest for n in notations):
            return self.currentClef  # can be ''

        for i, n in enumerate(notations):
            if n.isRest:
                continue
            if n.tiedPrev and self.currentClef and not self.clefChangeBetweenTiedNotes:
                return self.currentClef

            if self.history:
                distanceToLastChange = self.currentIndex + i - self.history[-1][0]
                distanceFactor = self.changeDistanceFactor.get(distanceToLastChange, 1.0)
            else:
                distanceFactor = 1.0

            for clef, evaluator in self.clefEvaluators.items():
                points = sum(evaluator(p) for p in n.pitches)
                if clef != self.currentClef:
                    a, b = clef, self.currentClef
                    pair = (a, b) if a < b else (b, a)
                    changeFactor = self.clefChangeFactor.get(pair, 1.0)
                    points *= changeFactor * distanceFactor
                else:
                    points *= self.biasFactor
                pointsPerClef[clef] = pointsPerClef.get(clef, 0) + points

        return max(pointsPerClef.items(), key=lambda pair: pair[1])[0]

    def process(self, notations: Sequence[Notation]) -> str:
        clef = self._bestClef(notations=notations)
        if not clef:
            assert not self.currentClef

        elif clef and not self.currentClef:
            if self.history:
                # Rewrite history
                idx, clefat0 = self.history[0]
                assert not clefat0
                self.history[0] = (idx, clef)

        self.history.append((self.currentIndex, clef))
        self.currentIndex += len(notations)
        self.currentClef = clef
        return clef


def bestClefForNotations(notations: Sequence[Notation]) -> str:
    """
    Find the best clef to apply to all notations without changes

    Args:
        notations: a seq. of Notations

    Returns:
        the most appropriate clef, as a string. Might be an empty
        string if no pitched notations are given

    """
    clefeval = ClefEvaluator()
    clef = clefeval.process(notations)
    return clef


def findBestClefs(notations: list[Notation],
                  firstClef='',
                  windowSize=1,
                  simplificationThreshold=0.,
                  biasFactor=1.5,
                  addClefs=True,
                  key='',
                  breakTies=False
                  ) -> list[tuple[int, str]]:
    """
    Given a list of notations, find the clef changes

    Args:
        notations: the notations.
        firstClef: the clef to start with
        windowSize: the size of the sliding window. The bigger the window, the
            broader the context considered.
        simplificationThreshold: if given, this is used to perform a simplification using
            the visvalingam-wyatt algorithm. The bigger this value, the
            sparser the clef changes. A value of 0. disables simpplification
        biasFactor: The higher this value, the more weight is given to the
            previous clef, thus making it more difficult to change clef
            more minor jumps
        addClefs: if True, add a Clef change (an attachment.Clef) to the
            notation where a clef change should happen.
        key: the property key to add to the notation to mark
             a clef change. Setting this property alone will not
             result in a clef change in the notation (see `addClefs`)
        breakTies: if True, a clef change is acceptable between tied notations


    Returns:
        a list of tuples (notationsindex, clef) where notationsindex is a
        list of indexes into the notations passed, indicating where a given
        clef should be applied, and clef is the clef to be applied to that
        notation. If addClefs=True, then these clefs are actually applied
        to the given notations. If key is given, a property with the given
        key is set to the name of the clef to set at that notation
    """
    points = []
    clefbyindex = ['treble15', 'treble8', 'treble', 'bass', 'bass8', 'bass15']
    notations = [n for n in notations if not n.isRest]
    if not notations:
        logger.debug("No pitched notations given")
        return []

    evaluator = ClefEvaluator(clefChangeBetweenTiedNotes=breakTies,
                              biasFactor=biasFactor,
                              changeDistanceFactor=None,
                              firstClef=firstClef)

    for i, group in enumerate(iterlib.window(notations, size=windowSize)):
        currentclef = evaluator.process(group)
        points.append((i, clefbyindex.index(currentclef)))

    if simplificationThreshold > 0 and len(points) > 2:
        import visvalingamwyatt
        simplifier = visvalingamwyatt.Simplifier(points)
        points = simplifier.simplify(threshold=simplificationThreshold)

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
        if key:
            n.setProperty(key, clef)
        if addClefs:
            n.addAttachment(attachment.Clef(clef))
    return clefs


def bestClefForPitch(pitch: float,
                     clefs: Sequence[str],
                     evaluators: dict[str, bpf4.BpfInterface] | None = None
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

    if all(n.isRest for n in notations):
        return [('treble', notations)]

    if maxstaves == 1:
        clef = bestClefForNotations(notations)
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
            ('treble15', 'treble', 'bass15'),
            ('treble15', 'bass', 'bass8'),

            ('treble', 'bass', 'bass8'),
            ('treble', 'bass', 'bass15')
        ]
    elif maxstaves == 4:
        possibleClefs = [
            ('treble15', 'treble', 'bass', 'bass8'),
            ('treble15', 'treble', 'bass', 'bass15'),

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
        if n.isRest:
            for part in parts.values():
                # part.append(n.copy())
                part.append(n)
        else:
            pitchindexToClef = [n.getClefHint(i) or bestClefForPitch(p, clefs=clefs, evaluators=evaluators)[0]
                                for i, p in enumerate(n.pitches)]
            clef0 = pitchindexToClef[0]
            if len(n.pitches) == 1 or all(clef == clef0 for clef in pitchindexToClef):
                parts[clef0].append(n)
                for otherclef in clefs:
                    if otherclef != clef0 and n.duration > 0:
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
