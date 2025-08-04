from __future__ import annotations

import itertools
from functools import cache
from emlib import iterlib
from pitchtools import n2m
from .common import logger
from . import attachment
from . import definitions
from .notation import Notation
    
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
    import bpf4
    from maelzel.scoring import spanner as _spanner


@cache
def clefFitnessCurves() -> dict[str, bpf4.BpfInterface]:
    """
    A dict {clef: bpf} where bpf maps midipitch to fitness

    Fitness indicates how good a fit this pitch is for this clef,
    based purely on the pitch

    Returns:
        a dict {clef(str): bpf}
    """
    import bpf4
    return {
        'treble15': bpf4.linear(
            (n2m("5C"), -2),
            (n2m("6C"), 0),
            (n2m("6A"), 1),
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
            (n2m("6G"), 0.8),
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


class ClefChangesEvaluator:
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
        firstClef: if given, use this clef as initial clef
        possibleClefs: if given, a selection of possible clefs
    """
    # We never autoevaluate to alto clef, but alto can still be set manually
    clefs = ['treble15', 'treble8', 'treble', 'bass', 'bass8', 'bass15']

    clefChangeFactor = {
        ('treble', 'treble8'): 0.5,
        ('treble8', 'treble15'): 0.2,
        ('treble', 'treble15'): 0.9,
        ('bass', 'treble15'): 0.9,
        ('bass', 'bass8'): 0.5,
        ('bass', 'bass15'): 0.9,
        ('bass8', 'bass15'): 0.2,
    }

    clefPriorityFactors = {
        'treble': 1.0,
        'bass': 1.0,
        'treble15': 0.9,
        'bass15': 0.9,
        'treble8': 0.7,
        'bass8': 0.7,
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
                 possibleClefs: list[str] | None = None,
                 firstClef=''):
        clefCurves = clefFitnessCurves()
        if possibleClefs is not None:
            clefCurves = {clef: curve for clef, curve in clefCurves.items()
                          if clef in possibleClefs}
        self.biasFactor = biasFactor
        self.clefChangeBetweenTiedNotes = clefChangeBetweenTiedNotes
        self.clefCurves = clefCurves
        self.history: list[tuple[int, str, float]] = []
        self.currentClef = firstClef
        self.currentIndex = 0
        self.changeDistanceFactor = changeDistanceFactor or ClefChangesEvaluator.defaultDistanceFactor

    def _bestClef(self, notations: Sequence[Notation]) -> tuple[str, float]:
        pointsPerClef: dict[str, float] = {}
        if all(n.isRest for n in notations):
            return self.currentClef, 0  # can be ''

        for i, n in enumerate(notations):
            if n.isRest:
                continue
            if n.tiedPrev and self.currentClef and not self.clefChangeBetweenTiedNotes:
                return self.currentClef, 0

            for clef, curve in self.clefCurves.items():
                points = sum(curve(p) for p in n.pitches)
                if clef != self.currentClef:
                    if self.history:
                        lastChange = next((idx0 for idx0, clef0, pts in reversed(self.history) if clef0 != clef), 0)
                        distanceToLastChange = self.currentIndex + i - lastChange
                        distanceFactor = self.changeDistanceFactor.get(distanceToLastChange, 1.0)
                    else:
                        distanceFactor = 1.0
                    a, b = clef, self.currentClef
                    pair = (a, b) if a < b else (b, a)
                    changeFactor = self.clefChangeFactor.get(pair, 1.0)
                    points *= changeFactor * distanceFactor
                else:
                    points *= self.biasFactor
                pointsPerClef[clef] = pointsPerClef.get(clef, 0) + points

        clef, points = max(pointsPerClef.items(), key=lambda pair: pair[1])
        return clef, points

    def process(self, notations: Sequence[Notation]) -> tuple[str, float]:
        clef, points = self._bestClef(notations=notations)
        if not clef:
            assert not self.currentClef

        elif clef and not self.currentClef:
            if self.history:
                # Rewrite history
                idx, clefat0, points0 = self.history[0]
                assert not clefat0
                self.history[0] = (idx, clef, points + points0 )

        self.history.append((self.currentIndex, clef, points))
        self.currentIndex += len(notations)
        self.currentClef = clef
        return clef, points


def bestClefForNotations(notations: Sequence[Notation],
                         possibleClefs: list[str] | None = None,
                         windowSize=4
                         ) -> str:
    """
    Find the best clef to apply to all notations without changes

    Args:
        notations: a seq. of Notations
        possibleClefs: if given, a list of possible clefs. Otherwise,
            all supported clefs are considered.
        windowSize: number of notations to analyze in a batch

    Returns:
        the most appropriate clef, as a string. Might be an empty
        string if no pitched notations are given

    """
    clefeval = ClefChangesEvaluator(possibleClefs=possibleClefs)
    clefhistory = {}
    for group in itertools.batched(notations, windowSize):
        clef, points = clefeval.process(group)
        clefhistory[clef] = clefhistory.setdefault(clef, 0) + points
    bestclef, points = max(clefhistory.items(), key=lambda pair: pair[1])
    return bestclef


def findBestClefs(notations: list[Notation],
                  firstClef='',
                  windowSize=1,
                  simplificationThreshold=0.,
                  biasFactor=1.5,
                  addClefs=True,
                  key='',
                  breakTies=False,
                  possibleClefs: list[str] | None = None,
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
        possibleClefs: if given, a list of possible clefs


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

    evaluator = ClefChangesEvaluator(clefChangeBetweenTiedNotes=breakTies,
                                     biasFactor=biasFactor,
                                     changeDistanceFactor=None,
                                     firstClef=firstClef,
                                     possibleClefs=possibleClefs)

    for i, group in enumerate(iterlib.window(notations, size=windowSize)):
        currentclef, _ = evaluator.process(group)
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


class SimpleClefEvaluator:
    """
    A stateless (context-free) clef evaluator
    
    It determines the best clef for a pitch solely based on its
    pitch
    """
    _cache: dict[None | tuple[str, ...], SimpleClefEvaluator] = {}

    def __new__(cls, clefs: tuple[str, ...] | None = None):
        if clefs in cls._cache:
            return cls._cache[clefs]
        return super().__new__(cls)

    def __init__(self, clefs: tuple[str, ...] | None = None):
        clefcurves = clefFitnessCurves()
        if clefs is not None:
            clefcurves = {clef: curve for clef, curve in clefcurves.items() if clef in clefs}
        self.curves = clefcurves
        self.clefs = list(clefcurves.keys())
        self._cache[clefs] = self

    def __call__(self, pitch: float) -> tuple[str, float]:
        points, clef = max((curve(pitch), clef) for clef, curve in self.curves.items())
        return clef, points
    
    def process(self, pitches: list[float]) -> tuple[str, float]:
        """
        Find the best clef for a sequence of pitches as a whole
        
        Args:
            pitches: 

        Returns:
            a tuple (clef: str, points: float)
        """
        clefs = {}
        for pitch in pitches:
            clef, points = self(pitch)
            clefs[clef] = clefs.setdefault(clef, 0) + points
        points, clef = max((points, clef) for clef, points in clefs.items())
        return clef, points


def bestClefForPitch(pitch: float,
                     clefs: Sequence[str] | None = None,
                     ) -> tuple[str, float]:
    """
    Determines the most appropriate clef for the given pitch

    If using repeatedly, use SimpleClefEvaluator instead.

    Args:
        pitch: the pitch
        clefs: the clefs to considere

    Returns:
        a tuple (clef: str, fitness: float), where clef is the name of the
        clef ('treble', 'bass', 'treble8', etc) and fitness indicates how good
        this clef is for representing this pitch (the higher, the more adequate
        the clef is)

    """
    return SimpleClefEvaluator(clefs)(pitch)


def _groupNotations(ns: list[Notation], groupingSpanners: tuple[_spanner.Spanner, ...] = ()) -> list[Notation | list[Notation]]:
    group = []
    groupid = ''
    out = []
    for n in ns:
        if groupid:
            group.append(n)
            if n.spanners and any(sp.kind == 'end' and sp.uuid == groupid for sp in n.spanners):
                out.append(group.copy())
                groupid = ''
                group.clear()
        else:
            if n.spanners and (sp:=next((sp for sp in n.spanners
                                        if sp.kind == 'start' and (not groupingSpanners or isinstance(sp, groupingSpanners))), None)):
                groupid = sp.uuid
                assert len(group) == 0
                group.append(n)
            else:
                out.append(n)
    return out


def explodeNotations(notations: list[Notation],
                     maxstaves=3,
                     ) -> list[tuple[str, list[Notation]]]:
    """
    Distribute notations across different clefs

    Args:
        notations: the notations to distribute
        maxstaves: max. number of staves

    Returns:
        a list of pairs (clefname: str, notations: list[Notation])
    """

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

    results = {}
    for clefs in possibleClefs:
        clefeval = ClefChangesEvaluator(possibleClefs=list(clefs))
        totalFitness = 0
        # for n in notations:
        for group in _groupNotations(notations):
            # clef, points = clefeval.process([n])
            clef, points = clefeval.process(group if isinstance(group, list) else [group])
            totalFitness += points
        results[clefs] = totalFitness

    bestClefCombination = max(results.items(), key=lambda pair: pair[1])[0]
    return splitNotationsByClef(notations, clefs=bestClefCombination)


def splitNotationsByClef(notations: list[Notation],
                         clefs: Sequence[str],
                         groupNotationsInSpanners=True
                         ) -> list[tuple[str, list[Notation]]]:
    """
    Split the given notations across different clefs

    Whenever a notation is assigned to a given clef, the other
    parts are filled with a rest of the same duration

    Args:
        notations: notations to split
        clefs: clefs for each part

    Returns:
        a list of pairs (clefname: str, notations: list[Notation])
    """
    if any(clef not in definitions.clefs for clef in clefs):
        clef = next(clef for clef in clefs if clef not in definitions.clefs)
        raise ValueError(f"Clef {clef} not known. Expected {definitions.clefs.keys()}")

    # Normalize clefs
    clefs = [definitions.clefs[clef] for clef in clefs]
    clefs = sorted(clefs, key=lambda clef: definitions.clefSortOrder[clef])
    parts = {clef: [] for clef in clefs}
    lastn = len(notations) - 1
    if not isinstance(clefs, tuple):
        clefs = tuple(clefs)
    clefeval = SimpleClefEvaluator(clefs=clefs)
    
    def handleNotation(n: Notation, nidx: int) -> None:
        if n.isRest:
            for part in parts.values():
                # should it be a copy?
                part.append(n)
        else:
            pitchindexToClef = [n._getClefHint(i) or clefeval(p)[0]
                                for i, p in enumerate(n.pitches)]
            clef0 = pitchindexToClef[0]
            if len(n.pitches) == 1 or all(clef == clef0 for clef in pitchindexToClef):
                parts[clef0].append(n)
                if n.duration > 0:
                    for otherclef in clefs:
                        if otherclef != clef0:
                            parts[otherclef].append(n.asRest())
            else:
                # A chord, distribute notes within parts
                for clef in clefs:
                    indexes = [i for i, clef2 in enumerate(pitchindexToClef) if clef == clef2]
                    if indexes:
                        parts[clef].append(n.extractPartialNotation(indexes))
                        if n.gliss and nidx < lastn:
                            for idx in indexes:
                                notations[nidx+1]._setClefHint(clef, idx)
                    elif n.duration > 0:
                        parts[clef].append(n.asRest())
            
    if groupNotationsInSpanners:
        nidx = 0
        for item in _groupNotations(notations):
            if isinstance(item, Notation):
                handleNotation(item, nidx)
                nidx += 1
            else:
                pitches = []
                for n in item:
                    pitches.extend(n.pitches)
                groupclef, _ = clefeval.process(pitches)
                nidx += len(item)
                parts[groupclef].extend(item)
                for clef, part in parts.items():
                    if clef != groupclef:
                        for n in item:
                            part.append(n.asRest())
    else:
        for nidx, n in enumerate(notations):
            handleNotation(n, nidx)           
                            
    return [(clef, part) for clef, part in parts.items()
            if part and not all(item.isRest for item in part)]
