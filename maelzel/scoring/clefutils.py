from __future__ import annotations

import itertools
from functools import cache
from emlib import iterlib
from pitchtools import n2m
from .common import logger
from . import attachment
from . import definitions
from .notation import Notation
from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
    import bpf4
    from maelzel.scoring import spanner as _spanner


@dataclass
class _ClefDefinition:
    """
    Defines a clef to evaluate clef changes
    """
    name: str
    """Name of the clef"""
    
    center: float
    """Center of gravity of the clef, used for sorting"""
    
    fitness: bpf4.BpfInterface
    """Maps pitch to fitness within this clef"""

    @property
    def transposing(self) -> bool:
        last = self.name[-1]
        return last == "8" or last == "5"
        
    
@cache
def clefDefinitions() -> dict[str, _ClefDefinition]:
    """
    A dict {clef: clefdef}

    Fitness indicates how good a fit a pitch is for the given clef,
    based purely on the pitch

    Returns:
        a dict {clef(str): bpf}
    """
    import bpf4
    return {
        'treble15': _ClefDefinition('treble15', center=n2m("7F"), fitness=bpf4.linear(
            (n2m("4C"), -100),
            (n2m("5C"), -2),
            (n2m("6C"), 0),
            (n2m("6A"), 1),
            (n2m("10C"), 1))),
        'treble8': _ClefDefinition('treble8', center=n2m("6D"), fitness=bpf4.linear(
            (n2m("3C"), -100),
            (n2m("4G"), -2),
            (n2m("5C"), 0),
            (n2m("5F"), 1),
            (n2m("7C"), 1),
            (n2m("7E"), 0))),

        'treble': _ClefDefinition('treble', center=n2m("4B"), fitness=bpf4.linear(
            (0, -100),
            (n2m("3C"), -10),
            (n2m("3E"), 0),
            (n2m("3G"), 0.8),
            (n2m("3B"), 1),
            (n2m("6F"), 1),
            (n2m("6G"), 0.8),
            (n2m("6B"), 0),
            (n2m("7A"), -2))),

        'bass': _ClefDefinition('bass', center=n2m("3E"), fitness=bpf4.linear(
            (n2m("0A"), -10),
            (n2m("1E"), 0),
            (n2m("1B"), 1),
            (n2m("4D"), 1),
            (n2m("4F"), 0),
            (n2m("4B"), -10),
            (n2m("5G"), -100) )),

        'bass8': _ClefDefinition('bass8', center=n2m("2D"), fitness=bpf4.linear(
            (n2m("-1A"), -2),
            (n2m("0C"), 0),
            (n2m("0E"), 1),
            (n2m("2A"), 1),
            (n2m("3C"), 0),
            (n2m("3F"), -10),
            (n2m("3A"), -100))),

        'bass15': _ClefDefinition('bass15', center=n2m("1F"), fitness=bpf4.linear(
            (n2m("-1D"), 1),
            (n2m("1C"), 1),
            (n2m("1G"), 0),
            (n2m("2C"), -2),
            (n2m("2A"), -100)))
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
        transposingFactor: factor applied to the fitness score of a clef
            if it is a transposing clef
    """
    # We never autoevaluate to alto clef, but alto can still be set manually
    clefs = ['treble15', 'treble8', 'treble', 'bass', 'bass8', 'bass15']

    # Pairs need to be sorted alphabetically ('bass' < 'treble')
    clefChangeFactor = {
        ('treble', 'treble8'): 0.2,   # discourage octave changes in preference of double octaves
        ('treble15', 'treble8'): 0.0, # do not allow 8va to 15a and viceversa
        ('treble', 'treble15'): 0.9,
        ('bass', 'bass8'): 0.2,
        ('bass', 'bass15'): 0.9,
        ('bass15', 'bass8'): 0.0,  # do not allow 8va to 15a and viceversa
        # ('bass8', 'treble8'): 0.5,
        ('bass8', 'treble15'): 0.2,
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
                 possibleClefs: Sequence[str] | None = None,
                 transposingFactor: float = 1.0,
                 firstClef=''):
        clefdefs = clefDefinitions()
        if possibleClefs is not None:
            clefdefs = {clef: curve for clef, curve in clefdefs.items()
                          if clef in possibleClefs}
        self.biasFactor = biasFactor
        self.clefChangeBetweenTiedNotes = clefChangeBetweenTiedNotes
        self.clefDefinitions: dict[str, _ClefDefinition] = clefdefs
        self.history: list[tuple[int, str, float]] = []
        self.currentClef = firstClef
        self.currentIndex = 0
        self.transposingFactor = transposingFactor
        self.changeDistanceFactor = changeDistanceFactor or ClefChangesEvaluator.defaultDistanceFactor

    def _bestClef(self, notations: Sequence[Notation]) -> tuple[str, float]:
        if all(n.isRest for n in notations):
            return self.currentClef, 0  # can be ''

        pointsPerClef: dict[str, float] = {}
        
        for i, n in enumerate(notations):
            if n.isRest:
                continue
            if n.tiedPrev and self.currentClef and not self.clefChangeBetweenTiedNotes:
                return self.currentClef, 0

            for clef, clefdef in self.clefDefinitions.items():
                points = sum(clefdef.fitness(p) for p in n.pitches)
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
                    if points > 0:
                        # changeFactor should attenuate only possitive points, not
                        # penalties
                        points *= changeFactor * distanceFactor
                else:
                    points *= self.biasFactor
                if clefdef.transposing:
                    points *= self.transposingFactor
                pointsPerClef[clef] = pointsPerClef.get(clef, 0) + points

        clef, points = max(pointsPerClef.items(), key=lambda pair: pair[1])
        return clef, points

    def process(self, notations: Sequence[Notation], advance: int = 0) -> tuple[str, float]:
        """
        Process the given notations, return the best clef
        
        Args:
            notations: the notations to evaluate
            advance: how many steps to advance the clef. This is relevant to 
                count the distance between clef changes. If not given, we advance
                by the number of notations passed. It can be less than that
                if using a windowed approach with overlap between notations

        Returns:
            a tuple (clef, points), where clef is the best clef for the
            given notations, based on both the notations and the history
        """
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
        self.currentIndex += advance or len(notations) 
        self.currentClef = clef
        return clef, points


def bestClefForNotations(notations: Sequence[Notation],
                         possibleClefs: list[str] | None = None,
                         windowSize=4,
                         hopSize=0
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
    if all(n.isRest for n in notations):
        return ''
    
    clefeval = ClefChangesEvaluator(possibleClefs=possibleClefs)
    clefhistory = {}
    if hopSize == 0:
        hopSize = windowSize
    else:
        assert hopSize <= windowSize
        
    for group in itertools.batched(notations, windowSize):
        clef, points = clefeval.process(group, advance=hopSize)
        clefhistory[clef] = clefhistory.setdefault(clef, 0) + points
    bestclef, points = max(clefhistory.items(), key=lambda pair: pair[1])
    return bestclef


def clefChanges(notations: list[Notation],
                firstClef='',
                windowSize=1,
                simplificationThreshold=0.,
                biasFactor=1.5,
                apply=True,
                key='',
                breakTies=False,
                possibleClefs: Sequence[str] | None = None,
                maxClef='',
                minClef='',
                transposingFactor=0.75,
                ) -> list[tuple[int, str]]:
    """
    Find the most adequate clef changes for the given notations

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
        apply: if True, add a Clef change (an attachment.Clef) to the
            notation where a clef change should happen.
        key: the property key to add to the notation to mark
             a clef change. Setting this property alone will not
             result in a clef change in the notation (see `apply`)
        breakTies: if True, a clef change is acceptable between tied notations
        possibleClefs: if given, a list of possible clefs
        minClef: if given, only use clefs equal or higher to minClef
        maxClef: if given, only use clefs equal or lower to maxClef


    Returns:
        a list of tuples (notationsindex: int, clef: str) where notationsindex
        is the index of the notation where a given clef should be applied,
        and clef is the clef to be applied to that notation. If addClefs=True,
        then these clefs are actually applied to the given notations. If key
        is given, a property with the given key is set to the name of the clef
        to set at that notation
    """
    clefdefs = clefDefinitions()
    if not possibleClefs:
        possibleClefs = list(clefdefs.keys())
    if minClef:
        center = clefdefs[minClef].center
        possibleClefs = [clef for clef in possibleClefs if clefdefs[clef].center >= center]
    if maxClef:
        center = clefdefs[maxClef].center
        possibleClefs = [clef for clef in possibleClefs if clefdefs[clef].center <= center]
        
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
                                     possibleClefs=possibleClefs,
                                     transposingFactor=transposingFactor)

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
    if key or apply:
        for idx, clef in clefs:
            n = notations[idx]
            if key:
                n.setProperty(key, clef)
            if apply:
                n.addAttachment(attachment.Clef(clef))
    return clefs


class SimpleClefEvaluator:
    """
    A stateless (context-free) clef evaluator
    
    It determines the best clef for a pitch solely based on its
    pitch
    """
    _cache: dict[None | tuple[str, ...], SimpleClefEvaluator] = {}

    def __new__(cls, clefs: Sequence[str] | None = None):
        if clefs is not None and not isinstance(clefs, tuple):
            clefs = tuple(sorted(clefs))
        if clefs in cls._cache:
            return cls._cache[clefs]
        return super().__new__(cls)

    def __init__(self, clefs: Sequence[str] | None = None):
        clefcurves = clefDefinitions()
        if clefs is not None:
            if not isinstance(clefs, tuple):
                clefs = tuple(sorted(clefs))
            clefcurves = {clef: curve for clef, curve in clefcurves.items() if clef in clefs}
        self.curves = clefcurves
        self.clefs = list(clefcurves.keys())
        self._cache[clefs] = self

    def __call__(self, pitch: float) -> tuple[str, float]:
        points, clef = max((curve.fitness(pitch), clef) for clef, curve in self.curves.items())
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


def _groupNotationsBySpanner(ns: Sequence[Notation]) -> dict[_spanner.Spanner, list[Notation]]:
    openspanners: set[str] = set()
    groups: dict[str, list[Notation]] = {}
    uuidToSpanner: dict[str, _spanner.Spanner] = {}
    for n in ns:
        if openspanners:
            for uuid in openspanners:
                groups[uuid].append(n)
        if n.spanners:
            for spanner in n.spanners:
                if spanner.kind == 'start':
                    openspanners.add(spanner.uuid)
                    uuidToSpanner[spanner.uuid] = spanner
                    groups[spanner.uuid] = [n]
                else:
                    if spanner.uuid in openspanners:
                        openspanners.remove(spanner.uuid)
                        groups[spanner.uuid].append(n)
    return {uuidToSpanner[uuid]: notations for uuid, notations in groups.items()}


@cache
def _clefCombinations(minStaves: int, maxStaves: int, exclude: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    allcombinations = []
    for num in range(minStaves, maxStaves+1):
        combinations = _possibleCombinations(numStaves=num, exclude=exclude)
        allcombinations.extend(combinations)
    return allcombinations


@cache
def _possibleCombinations(numStaves: int, exclude: tuple[str, ...] = ()
                          ) -> list[tuple[str, ...]]:
    """
    Possible combinations of clefs, given the number of staves
    
    Args:
        numStaves: number of different staves 
        exclude: tuple with clefs which should be excluded

    Returns:
        a list of combinations, where each item is a tuple of different clefs
    """
    if numStaves == 1:
        combinations = [('treble15',), ('treble8',), ('treble',), ('bass',), ('bass8',), ('bass15',)]
    elif numStaves == 2:
        combinations = [
            ('treble15', 'treble'),
            ('treble15', 'bass'),
            ('treble8', 'bass'),
            ('treble8', 'bass8'),
            ('treble', 'bass'),
            ('treble', 'bass8')]
    elif numStaves == 3:
        combinations = [
            ('treble15', 'treble', 'bass'),
            ('treble15', 'treble', 'bass8'),
            ('treble15', 'treble', 'bass15'),
            ('treble15', 'bass', 'bass8'),
            ('treble', 'bass', 'bass8'),
            ('treble', 'bass', 'bass15')]
    elif numStaves == 4:
        combinations = [
            ('treble15', 'treble', 'bass', 'bass8'),
            ('treble15', 'treble', 'bass', 'bass15')]
    else:
        raise ValueError(f"Invalid numstaves: {numStaves}")
    if exclude:
        combinations = [comb for comb in combinations if all(clef not in comb for clef in exclude)]
    return combinations
    

def bestClefCombination(notations: list[Notation],
                        maxStaves: int,
                        minStaves: int = 1,
                        singleStaffRange=12,
                        groupNotationsWithinSpanner=False,
                        staffPenalty=1.2,
                        transposingPenalty=1.3
                        ) -> tuple[str, ...]:
    """
    Find the best combination of clefs for the given notations

    Args:
        notations: the notations to distribute among staves with different clefs
        maxStaves: max. number of staves
        minStaves: min. number of staves
        singleStaffRange: if notations can be fit within this range no new
            staves are created
        groupNotationsWithinSpanner: assign the same staff to notations within a spanner
        staffPenalty: penalty applied to the creation of a staff
        transposingPenalty: penalty applied when choosing a transposing clef

    Returns:
        a tuple of clefs
    """
    pitchedNotations = [n for n in notations if not n.isRest]

    if not pitchedNotations:
        return ('treble',)

    minpitch = min(n.pitchRange()[0] for n in pitchedNotations)
    maxpitch = max(n.pitchRange()[1] for n in pitchedNotations)

    if maxStaves == 1 or maxpitch - minpitch <= singleStaffRange:
        return (bestClefForNotations(notations),)

    possibleCombinations = _clefCombinations(maxStaves=maxStaves, minStaves=minStaves)
    results: dict[tuple[str, ...], float] = {}
    if groupNotationsWithinSpanner:
        groups = _groupNotations(notations)
        for clefs in possibleCombinations:
            clefeval = SimpleClefEvaluator(clefs=clefs)
            fitness = 0.
            for group in groups:
                if isinstance(group, list):
                    pitches = sum((n.pitches for n in group), [])
                    fitness += clefeval.process(pitches)[1]
                else:
                    fitness += sum(clefeval(pitch)[1] for pitch in group.pitches)
            results[clefs] = fitness
    else:
        for clefs in possibleCombinations:
            clefeval = SimpleClefEvaluator(clefs=clefs)
            fitness = sum(clefeval(pitch)[1] for n in notations for pitch in n.pitches)
            results[clefs] = fitness
    for clefs, fitness in results.items():
        if len(clefs) > 1:
            fitness *= 1/((len(clefs) - 1) * staffPenalty)
        for clef in clefs:
            last = clef[-1]
            if last == "8" or last == "5":
                fitness *= 1/transposingPenalty
        results[clefs] = fitness
    out = max(results.items(), key=lambda pair: pair[1])[0]
    return out

def explodeNotations(notations: list[Notation],
                     maxStaves: int,
                     minStaves=0,
                     singleStaffRange=12,
                     distributeSpanners=True,
                     staffPenalty=1.2,
                     groupNotationsWithinSpanner=False
                     ) -> list[tuple[str, list[Notation]]]:
    """
    Distribute notations across different clefs

    Args:
        notations: the notations to distribute
        maxStaves: max. number of staves, 1 <= maxStaves <= 4
        singleStaffRange: if notations fit within this range,
            only one staff is used

    Returns:
        a list of pairs (clefname: str, notations: list[Notation]). It can be 
        less than the number of staves given, but not more.
    """
    if not minStaves:
        minStaves = maxStaves
        
    bestClefs = bestClefCombination(notations=notations,
                                    maxStaves=maxStaves,
                                    minStaves=minStaves,
                                    singleStaffRange=singleStaffRange,
                                    staffPenalty=staffPenalty,
                                    groupNotationsWithinSpanner=groupNotationsWithinSpanner)
    if len(bestClefs) == 1:
        return [(bestClefs[0], notations)]

    return splitNotationsByClef(notations,
                                clefs=bestClefs,
                                groupNotationsInSpanners=False,
                                distributeSpanners=distributeSpanners)


@cache
def clefsBetween(minclef='',
                 maxclef='',
                 includemin=True,
                 includemax=True,
                 possibleClefs: tuple[str, ...] = (),
                 excludeClefs: tuple[str, ...] = ()
                 ) -> tuple[str, ...]:
    """
    Returns the clefs between the given bounds

    Args:
        minclef: the lowest clef to include
        maxclef: the highest clef to include
        includemin: should the minimum clef be included?
        includemax: should the maximum clef be included?
        possibleClefs: clefs to choose from. If not given, all possible clefs
            are used

    Returns:

    """
    if not (minclef or maxclef):
        raise ValueError("At least minclef or maxclef should be given")

    if not possibleClefs:
        possibleClefs = definitions.clefsByOrder

    if excludeClefs:
        possibleClefs = tuple(clef for clef in possibleClefs if clef not in excludeClefs)

    if not minclef:
        minclef = definitions.clefsByOrder[0]
    if not maxclef:
        maxclef = definitions.clefsByOrder[-1]

    minorder = definitions.clefSortOrder[minclef]
    maxorder = definitions.clefSortOrder[maxclef]

    if includemin and includemax:
        clefs = [clef for clef in possibleClefs
                 if minorder <= definitions.clefSortOrder[clef] <= maxorder]
    elif includemin:
        clefs = [clef for clef in possibleClefs
                 if minorder <= definitions.clefSortOrder[clef] < maxorder]
    elif includemax:
        clefs = [clef for clef in possibleClefs
                 if minorder < definitions.clefSortOrder[clef] <= maxorder]
    else:
        clefs = [clef for clef in possibleClefs
                 if minorder < definitions.clefSortOrder[clef] < maxorder]
    return tuple(clefs)


def splitNotationsByClef(notations: list[Notation],
                         clefs: Sequence[str],
                         groupNotationsInSpanners=False,
                         distributeSpanners=True
                         ) -> list[tuple[str, list[Notation]]]:
    """
    Split the given notations across different clefs

    Whenever a notation is assigned to a given clef, the other
    parts are filled with a rest of the same duration

    Args:
        notations: notations to split
        clefs: clefs for each part
        groupNotationsInSpanners: if True, force all parts of all notations within spanners
            to be on one voice
        distributeSpanners: if True and spanners extend over notations across multiple staves,
            assign those spanners to the first and last notation belonging to the spanner
            on each staff

    Returns:
        a list of pairs (clefname: str, notations: list[Notation])
    """

    if any(clef not in definitions.clefs for clef in clefs):
        clef = next(clef for clef in clefs if clef not in definitions.clefs)
        raise ValueError(f"Clef {clef} not known. Expected {definitions.clefs.keys()}")

    # Normalize clefs
    clefs = [definitions.clefs[clef] for clef in clefs]
    clefs = sorted(clefs, key=lambda clef: definitions.clefSortOrder[clef])
    parts: dict[str, list[Notation]] = {clef: [] for clef in clefs}
    lastn = len(notations) - 1
    if not isinstance(clefs, tuple):
        clefs = tuple(clefs)
    clefeval = SimpleClefEvaluator(clefs=clefs)

    def _distrNotation(n: Notation, nidx: int, parts: dict[str, list[Notation]], ns: Sequence[Notation]) -> None:
        if n.isRest:
            for part in parts.values():
                part.append(n)  # should it be a copy?
            return
        pitchindexToClef: list[str] = [n._getClefHint(i) or clefeval(p)[0]
                                       for i, p in enumerate(n.pitches)]
        clef0 = pitchindexToClef[0]
        if len(n.pitches) == 1 or all(clef == clef0 for clef in pitchindexToClef):
            parts[clef0].append(n.copy(spanners=False))
            if n.duration > 0:
                for otherclef in clefs:
                    if otherclef != clef0:
                        parts[otherclef].append(n.asRest())
        else:
            # A chord, distribute notes within parts
            for clef in clefs:
                indexes = [i for i, clef2 in enumerate(pitchindexToClef) if clef == clef2]
                if indexes:
                    partialn = n.extractPartialNotation(indexes, spanners=False)
                    parts[clef].append(partialn)
                    if n.gliss and nidx < lastn:
                        for idx in indexes:
                            ns[nidx+1]._setClefHint(clef, idx)
                elif n.duration > 0:
                    parts[clef].append(n.asRest())

    prefix = "_spanner_"
    notationsBySpanner = None
    if distributeSpanners:
        notationsBySpanner = _groupNotationsBySpanner(notations)
        for spanner, notationsInSpanner in notationsBySpanner.items():
            for n in notationsInSpanner:
                n.setProperty(f"{prefix}{spanner.uuid}", True)
    
    if groupNotationsInSpanners:
        nidx = 0
        for item in _groupNotations(notations):
            if isinstance(item, Notation):
                _distrNotation(item, nidx, parts, notations)
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
            _distrNotation(n, nidx, parts, notations)

    parts = {clef: part for clef, part in parts.items()
             if part and not all(n.isRest for n in part)}

    if distributeSpanners:
        assert isinstance(notationsBySpanner, dict)
        uuidToSpanner = {spanner.uuid: spanner for spanner in notationsBySpanner}
        for part in parts.values():
            spannergroups: dict[str, list[Notation]] = {}
            for n in part:
                if n.properties:
                    for prop in n.properties:
                        if prop.startswith(prefix):
                            uuid = prop[len(prefix):]
                            spannergroups.setdefault(uuid, []).append(n)
            for uuid, notations in spannergroups.items():
                if len(notations) > 1:
                    spanner = uuidToSpanner[uuid]
                    notations[0].addSpanner(spanner, end=notations[-1])

    pairs = list(parts.items())
    pairs.sort(key=lambda pair: definitions.clefSortOrder[pair[0]])
    return pairs
