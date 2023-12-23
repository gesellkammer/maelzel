"""
Find the best enharmonic spelling for a sequence of notes
"""
from __future__ import annotations
from collections import defaultdict
from statistics import stdev
import functools
from dataclasses import dataclass, astuple as _astuple
from collections import deque
from math import sqrt
from .common import logger

from emlib import iterlib
import pitchtools as pt

from .notation import Notation

from typing import Sequence

MAXPENALTY = 999999

_quarternote_slotnames_up = ('C', 'C+', 'C#',
                             'C#+', 'D', 'D+', 'D#',
                             'D#+', 'E', 'E+',
                             'F', 'F+', 'F#',
                             'F#+', 'G', 'G+', 'G#',
                             'G#+', 'A', 'A+', 'A#',
                             'A#+', 'B', 'B+', 'C')

_quarternote_slotnames_down = ('C', 'Db-', 'Db',
                               'D-', 'D', 'Eb-', 'Eb',
                               'E-', 'E', 'F-',
                               'F', 'Gb-', 'Gb',
                               'G-', 'G', 'Ab-', 'Ab',
                               'A-', 'A', 'Bb-', 'Bb',
                               'B-', 'B', 'C-', 'C')

_quarternote_slotnames = ('C', 'C+', 'C#',
                          'D-', 'D', 'D+', 'D#',
                          'E-', 'E', 'E+',
                          'F', 'F+', 'F#',
                          'G-', 'G', 'G+', 'G#',
                          'A-', 'A', 'A+', 'Bb',
                          'B-', 'B', 'B+', 'C')

_semitone_slotnames_up = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C')
_semitone_slotnames_down = ('C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C')


def _reprslots(slots: dict[int, int], semitoneDivs=2) -> str:
    slotnames = []
    if semitoneDivs == 2:
        for slotnum, direction in slots.items():
            if direction == -1:
                slotname = _quarternote_slotnames_down[slotnum]
                slotnames.append(slotname)
            elif direction == 1:
                slotname = _quarternote_slotnames_up[slotnum]
                slotnames.append(slotname)
    elif semitoneDivs == 1:
        for slotnum, direction in slots.items():
            if direction == -1:
                slotname = _semitone_slotnames_down[slotnum]
                slotnames.append(slotname)
            elif direction == 1:
                slotname = _semitone_slotnames_up[slotnum]
                slotnames.append(slotname)
    else:
        raise ValueError("Only 1 or 2 semitone divisions are supported")
    return ', '.join(slotnames)


@dataclass
class EnharmonicOptions:

    debug: bool = False
    """Display debug information"""

    groupSize: int = 5
    """Max size of a group to be evaluated for best enharmonic variant"""

    groupStep: int = 2

    fixedSlotMaxHistory: int = 4

    threeQuarterMicrotonePenalty: float = 100.
    """penalty applied for 3/4 microtones like #+ or b- (prefer A+ over Bb-)"""

    respellingPenalty: float = 75.
    """penalty for respelling a pitch (C+ / Db-)"""

    confusingIntervalPenalty: float = 21.

    mispelledInterval: float = 60.

    unisonAlterationPenalty: float = 12.
    "penalty for C / C+"

    chordUnisonAlterationPenalty: float = 40.
    "penalty for C / C+ within a chord"

    chordMispelledInterval: float = 20.
    "penalty for D - Gb within a chord"

    intervalPenaltyWeight: float = 1.
    "The weight of the horizontal intervals, in pairs"

    groupPenaltyWeight: float = 2.
    "The weight of an enharmonic variant as a group (horizontally)"

    horizontalWeight: float = 1.
    "Groups both intervalPenaltyWeight and groupPenaltyWeight"

    verticalWeight: float = 0.05
    "The weight of how a variant affects chords (vertically)"

    _hash: int = 0

    def __hash__(self):
        if not self._hash:
            self._hash = hash(tuple(_astuple(self)))
        return self._hash


_defaultEnharmonicOptions = EnharmonicOptions()


def isEnharmonicVariantValid(notes: list[str]) -> bool:
    """
    Is the enharmonic spelling in this list of notes valid?

    Args:
        notes: a list of notenames

    Returns:
        True if the spelling used for these notes is valid

    A valid variant needs to follow these rules:
    * if one pitch is lower than the next its vertical position should
      never be higher (never allow a sequence like 4Eb-, 4D+75)
    * the same should be valid in the opposite direction. This makes a
      sequence like 4D+75, 4Eb- invalid, since the first pitch is higher
      than the second but its vertical position is lower
    """
    n0 = notes[0]
    p0 = pt.n2m(n0)
    for n1 in notes[1:]:
        p1 = pt.n2m(n1)
        if p0 < p1 and pt.vertical_position(n0) > pt.vertical_position(n1):
            return False
        if p0 > p1 and pt.vertical_position(n0) < pt.vertical_position(n1):
            return False
        p0 = p1
        n0 = n1
    return True


def groupPenalty(notes: list[str], options: EnharmonicOptions = None) -> tuple[float, str]:
    """
    Evaluate the enharmonic variant as a group

    Args:
        notes: the list of pitches
        options: an instance of EnharmonicOptions or None to use default

    Returns:
        a tuple (penalty, penalty sources), where penalty is an abstract
        number where higher means less fit, and penalty sources is a comma
        separated string collecting all sources of penalty

    """
    if options is None:
        options = _defaultEnharmonicOptions

    total = 0
    penaltysources = []
    notated: list[pt.NotatedPitch] = [pt.notated_pitch(n) for n in notes]
    #alterations = [n.alteration_direction() for n in notated
    #               if n.is_black_key]

    # Penalize crammed unisons
    accidentalsPerPos = defaultdict(set)
    for n in notated:
        accidentalsPerPos[n.vertical_position].add(n.accidental_name)
    for pos, accidentals in accidentalsPerPos.items():
        # print(pt.vertical_position_to_note(pos), pos, count)
        count = len(accidentals)
        if count == 3:
            # C# C C+
            total += 101
        elif count == 4:
            total += 1000
        elif count > 5:
            total += 999999

    # Penalize respellings
    # This penalizes C+ .. Db-
    spellingsPerIndex = defaultdict(set)
    for n in notated:
        if n.is_black_key:
            spellingsPerIndex[n.chromatic_index].add(n.chromatic_name)
    for idx, spellings in spellingsPerIndex.items():
        if len(spellings) > 1:
            penaltysources.append("respellingPenalty")
            total += options.respellingPenalty

    # Penalize 1.5 microtones (#+, b-)
    for n in notated:
        if abs(n.diatonic_alteration) == 1.5:
            penaltysources.append("threeQuarterMicrotone")
            total += options.threeQuarterMicrotonePenalty

    # Penalize successive microtones going in different directions
    # as the intervals are difficult to read
    for n0, n1 in iterlib.window(notated, 2):
        if n1.vertical_position - n0.vertical_position > 1 and n1.diatonic_alteration == -1.5:
            # C Eb-
            penaltysources.append("confusingInterval C Eb-")
            total += options.confusingIntervalPenalty * 3
        if n0.chromatic_alteration != 0 and n1.chromatic_alteration != 0:
            if n0.alteration_direction(min_alteration=0.25) != n1.alteration_direction(0.25):
                if n0.diatonic_alteration == n1.diatonic_alteration:
                    total += options.confusingIntervalPenalty * 2
                else:
                    total += options.confusingIntervalPenalty
            elif abs(n0.diatonic_alteration) == 0.5 and abs(n1.diatonic_alteration) == 1.5:
                # Penalize things like E+ D#+, prefer F- E-
                total += options.confusingIntervalPenalty * 2
    for n0, n1, n2 in iterlib.window(notated, 3):
        dpos0, dpitch0 = pt.notated_interval(n0.fullname, n1.fullname)
        dpos1, dpitch1 = pt.notated_interval(n1.fullname, n2.fullname)
        dpitch0rounded = round(dpitch0*2)/2
        drpitch1rounded = round(dpitch1*2)/2
        if dpos0 == 3 and dpitch0rounded == 4.5 and dpos1 == 0 and drpitch1rounded == 0.5:
            # 4D 4G-
            penaltysources.append(f"confusingInterval {n0.fullname}/{n1.fullname}")
            total += options.confusingIntervalPenalty * 2
    return total, ", ".join(penaltysources)


def intervalsPenalty(notes: list[str], chord=False, options: EnharmonicOptions = None
                     ) -> tuple[float, str]:
    if options is None:
        options = _defaultEnharmonicOptions
    total = 0
    sources = []
    for n0, n1 in iterlib.window(notes, 2):
        penalty, source = intervalPenalty(n0, n1, chord=chord, options=options)
        total += penalty
        sources.append(source)
    sources = [s for s in sources if s]
    return total, ", ".join(sources)


@functools.cache
def intervalPenalty(n0: str, n1: str, chord=False, options: EnharmonicOptions = _defaultEnharmonicOptions
                    ) -> tuple[float, str]:
    """
    Rate the penalty of the interval between n0 and n1

    Args:
        n0: the first notename
        n1: the second notename
        chord: if True, evaluate how proper this interval is in the context
            of a chord rather than a melody
        options: penalty options

    Returns:
        a tuple (penalty, penalty sources)
    """
    if pt.n2m(n0) > pt.n2m(n1):
        n0, n1 = n1, n0
    dpos, dpitch = pt.notated_interval(n0, n1)
    assert dpitch >= 0
    if dpos < 0:
        # 4Eb- / 4D#
        return MAXPENALTY, "Really confusing interval"

    if dpitch == 0:
        if dpos == 0:
            # same pitch, no penalty
            return 0, ""
        return MAXPENALTY, "Respelled pitch"

    if (dpos > 0 and dpitch < 0) or (dpos < 0 < dpitch):
        # ex: 4C# 4Db- or 4E+ 4Fb
        return MAXPENALTY, "Inverse pitch"

    dpos7 = dpos % 7
    dpitch12 = dpitch % 12
    notated0 = pt.notated_pitch(n0)
    notated1 = pt.notated_pitch(n1)
    penalties = []
    mispelledInterval = options.chordMispelledInterval if chord else options.mispelledInterval
    _ = penalties.append

    if chord and notated0.vertical_position == notated1.vertical_position:
        _((options.chordUnisonAlterationPenalty * 3, 'crammed unison'))

    if ((notated0.diatonic_alteration <= -1 and notated1.diatonic_alteration >= 1) or
            (notated0.diatonic_alteration >= 1 and notated1.diatonic_alteration <= -1)):
        _((options.confusingIntervalPenalty * 1, "Accidentals in mixed directions"))
    if dpos == 0:
        if dpitch >= 1.5:
            # G / Gb- or Gb / G+
            _((options.confusingIntervalPenalty * 10, f"confusing interval"))
        # same position, different pitch
        elif notated0.diatonic_alteration == -1.5 and notated1.diatonic_alteration == 0:
            # 4Gb- / 4G
            _((options.confusingIntervalPenalty*10, "confusingInterval"))
        elif notated0.diatonic_alteration == 0 and notated1.diatonic_alteration == 1.5:
            # 4G / 4G#+
            _((options.confusingIntervalPenalty*10, "confusingIntervl"))
        elif notated0.chromatic_alteration < 0 and notated1.chromatic_alteration > 0:
            # 4E- 4E+
            _((options.unisonAlterationPenalty*2, "unisonAlteration"))
        elif notated0.chromatic_alteration != 0 and notated1.chromatic_alteration == 0:
            # 4E+ 4E
            if not chord:
                _((options.unisonAlterationPenalty, f"unison"))
            else:
                _((options.chordUnisonAlterationPenalty, f"unison"))
        else:
            _((options.unisonAlterationPenalty, "unisonAlteration"))
    elif dpos == 1:
        # 4C 4D#, 4G 4A#  -> better 4C 4Eb or 4G 4Bb
        if dpitch >= 3:
            _((mispelledInterval, "mispelled"))

    elif dpos7 == 2:
        if dpitch12 > 5:
            _((mispelledInterval, "mispelled"))
        elif dpitch12 >= 4.5:
            _((options.confusingIntervalPenalty, "Db/F+"))
        elif dpitch12 <= 1.5:
            # 4F# 4Ab- (dpitch=2)
            _((mispelledInterval*2, "mispelledInterval!"))
        elif dpitch12 <= 2:
            _((mispelledInterval, "mispelledInterval"))
    elif dpos7 == 3:
        # 4D 4Gb (dpitch=4)
        if dpitch12 < 3.5:
            # 4D# 4Gb- (dpitch=3)
            _((options.confusingIntervalPenalty*2, "confusing interval"))
        elif dpitch12 == 3.5 and abs(notated0.diatonic_alteration) == 1.5 or abs(notated1.diatonic_alteration) == 1.5:
            # 4D 4Gb-
            _((options.confusingIntervalPenalty * 10, "confusing"))
        elif dpitch12 <= 4:
            _((mispelledInterval, "mispelledInterval"))
        elif dpitch12 == 4.5:
            _((options.confusingIntervalPenalty*1.5, "D/G-"))
        elif dpitch12 >= 7:
            # 4Db 4G#
            _((mispelledInterval, "mispelledInterval"))
    elif dpos7 == 4:  # a 5th
        if dpitch12 >= 9:
            # 4Db 4A#
            _((mispelledInterval, "mispelledInterval"))
        elif dpitch12 <= 5:
            # 4D# 4Ab
            _((mispelledInterval, "mispelledInterval"))
    elif dpos7 == 5:  # a 6th
        # C# Ab (7) / Db B (10)
        if not 7 < dpitch12 < 10:
            _((mispelledInterval, "mispelled"))
    total = sum(p for p, s in penalties)
    source = ", ".join(s for p, s in penalties) + f"({n0}/{n1})"
    return total, source


def _weight(values, method='stdev'):
    if method == 'avg':
        avg = sum(values) / len(values)
        if avg == 0:
            return 0
        return 1/avg
    else:
        if len(values) < 2:
            return 1
        d = stdev(values)
        if d == 0:
            return 0
        return 0 if d == 0 else 1/(d**2)


def _rateEnharmonicVariations(group: Sequence[Notation],
                              spellings: list[list[str]],
                              anchors: list[int],
                              options: EnharmonicOptions
                              ) -> list[str]:
    intervalPenalties = [intervalsPenalty(spelling, options=options)[0]
                         for spelling in spellings]
    groupPenalties = [groupPenalty(spelling, options=options)[0]
                      for spelling in spellings]
    chordPenalties = [_chordPenalty(notations=group,
                                    spelling=spelling,
                                    anchors=anchors,
                                    options=options)
                      for spelling in spellings]

    weightIntervalPenalty = _weight(intervalPenalties)
    weightGroupPenalty = _weight(groupPenalties)
    weightChordPenalty = _weight(chordPenalties)
    solutions = []
    intervalPenaltyWeight = options.intervalPenaltyWeight * options.horizontalWeight
    groupPenaltyWeight = options.groupPenaltyWeight * options.horizontalWeight
    for i in range(len(spellings)):
        a = intervalPenalties[i]**2 * weightIntervalPenalty * intervalPenaltyWeight
        b = groupPenalties[i]**2 * weightGroupPenalty * groupPenaltyWeight
        c = chordPenalties[i]**2 * weightChordPenalty * options.verticalWeight
        penalty = sqrt(a+b+c)
        solutions.append((penalty, spellings[i], a, b, c))
    solutions.sort(key=lambda solution: solution[0])
    if options.debug:
        for s in solutions:
            penalty, spelling, a, b, c = s
            print(f'Spelling: {" ".join(spelling)}, penalty={penalty: .4g}, '
                  f'intervalPenalty={a:.4g}, groupPenalty={b:.4g}, '
                  f'chordPenalty={c:.4g}')
    return solutions[0][1]


def _chordPenalty(notations: Sequence[Notation],
                  spelling: list[str],
                  anchors: list[int],
                  options: EnharmonicOptions
                  ) -> float:
    totalChordPenalty = 0
    for i, notation in enumerate(notations):
        notes = notation.resolveNotenames()
        anchor = anchors[i]
        notes[anchor] = spelling[i]
        chordrating, sources = _rateChordSpelling(notes, options=options)
        totalChordPenalty += chordrating
    return totalChordPenalty


def enharmonicPenalty(notes: list[str], options: EnharmonicOptions
                      ) -> float:
    """
    Rate how bad this enharmonic variant is

    Args:
        notes: a group of notenames to evaluate
        options: options to configure the evaluation

    Returns:
        a penalty value. The magnitude of this value is meaningless and is only
        relevant in relation to another penalty

    """
    intervalpenalty, sources0 = intervalsPenalty(notes, options=options)
    grouppenalty, sources1 = groupPenalty(notes, options=options)
    total = sqrt(intervalpenalty**2 + grouppenalty**2)
    return total


class SpellingHistory:
    def __init__(self, itemHistory=4, pitchHistory=5, divsPerSemitone=2):
        self.itemHistory = itemHistory
        self.pitchHistory = pitchHistory
        self.dequelen = itemHistory
        self.deque: deque[list[pt.NotatedPitch]] = deque(maxlen=self.dequelen)
        self.divsPerSemitone = divsPerSemitone
        numslots = 12 * divsPerSemitone
        self.slots = {idx: 0 for idx in range(numslots)}
        self.refcount = {idx: 0 for idx in range(numslots)}

    def clear(self):
        self.deque.clear()
        numslots = 12 * self.divsPerSemitone
        self.slots = {idx: 0 for idx in range(numslots)}
        self.refcount = {idx: 0 for idx in range(numslots)}

    def add(self, items: list[str]) -> None:
        ns = [pt.notated_pitch(item) for item in items]
        numpitches = sum(len(item) for item in self.deque)
        if len(self.deque) == self.itemHistory or numpitches >= self.pitchHistory:
            evicted = self.deque.popleft()
            # an append will evict an item
            for n in evicted:
                idx = n.microtone_index()
                self.refcount[idx] -= 1
                if self.refcount[idx] == 0:
                    self.slots[idx] = 0

        for n, item in zip(ns, items):
            idx = n.microtone_index()
            direction = n.alteration_direction()
            previousDirection = self.slots[idx]
            if previousDirection and previousDirection != direction:
                logger.info(f"Spelling error with {item}, spelling already fixed "
                            f"(previous direction: {previousDirection}")
            self.slots[idx] = direction
            self.refcount[idx] += 1

        self.deque.append(ns)

    def currentSpelling(self, index: int) -> int:
        val = self.slots[index]
        if val > 0:
            return 1
        elif val < 0:
            return -1
        return 0

    def spellingOk(self, notename: str) -> bool:
        n = pt.notated_pitch(notename)
        index = n.microtone_index()
        currentSpelling = self.currentSpelling(index)
        return currentSpelling == 0 or currentSpelling == n.alteration_direction()

    def dump(self) -> None:
        def idx2name(idx):
            return pt.m2n(60 + idx / 2)[1:]

        fixed = {f"{idx}:{idx2name(idx)}": val for idx, val in self.slots.items()
                 if val != 0}
        print(fixed)

    def addNotation(self, notation: Notation):
        """
        Add a Notation to the history

        Args:
            notation: the notation to add
        """
        notenames = notation.resolveNotenames()
        self.add(notenames)


def _rateChordSpelling(notes: Sequence[str], options: EnharmonicOptions) -> tuple[float, str]:
    totalpenalty = 0.
    sources = []
    for a, b in iterlib.combinations(notes, 2):
        penalty, source = intervalPenalty(a, b, chord=True, options=options)
        totalpenalty += penalty
        sources.append(source)
    sourcestr = ":".join(sources)
    return totalpenalty, sourcestr


def _makeFixedSlots(fixedNotes: list[str], semitoneDivs=2) -> dict[int, int]:
    slots = {}
    for n in fixedNotes:
        parsed = pt.notated_pitch(n)
        if semitoneDivs == 1:
            slots[parsed.chromatic_index] = parsed.alteration_direction()
        elif semitoneDivs == 2:
            slots[parsed.microtone_index(semitone_divisions=semitoneDivs)] = parsed.alteration_direction(min_alteration=0.5)
        else:
            raise ValueError(f"semitoneDivs can be 1 or 2, got {semitoneDivs}")
    return slots


def bestChordSpelling(notes: Sequence[str], options: EnharmonicOptions = None
                      ) -> tuple[str, ...]:
    notes2 = notes if isinstance(notes, tuple) else tuple(notes)
    return _bestChordSpelling(notes2, options=options or _defaultEnharmonicOptions)


@functools.cache
def _bestChordSpelling(notes: tuple[str, ...], options: EnharmonicOptions
                       ) -> tuple[str, ...]:
    fixedNotes = []
    notelist = list(notes)
    for i, n in enumerate(notes):
        if n.endswith('!'):
            notelist[i] = n = n[:-1]
            fixedNotes.append(n)
    slots = _makeFixedSlots(fixedNotes, semitoneDivs=2) if fixedNotes else None
    variants = pt.enharmonic_variations(notelist, fixedslots=slots)
    if not variants:
        return notes
    variants.sort(key=lambda v: _rateChordSpelling(v, options)[0])
    return variants[0]


def pitchSpellings(n: Notation) -> tuple[str, ...]:
    """
    Returns the explict spelling or the most appropriate spelling for the given Notation

    Args:
        n: the notation

    Returns:
        a tuple of notenames representing the most appropriate spelling for the pitches
        in this notation

    """
    if len(n) == 1:
        return n.notename(0),
    notenames = [n.notename(idx, addExplicitMark=True) for idx in range(len(n))]
    spelling = bestChordSpelling(notenames)
    return spelling


def _notationNotename(n: Notation, idx=0) -> str:
    if fixed := n.getFixedNotename(idx):
        return fixed
    if len(n.pitches) == 0:
        return pt.m2n(n.pitches[0])
    bestspelling = bestChordSpelling(n.resolveNotenames())
    return bestspelling[idx]


def fixEnharmonicsInPlace(notations: list[Notation],
                          eraseFixedNotes=False,
                          options: EnharmonicOptions = _defaultEnharmonicOptions,
                          spellingHistory: SpellingHistory | None = None
                          ) -> None:
    """
    Finds the best enharmonic spelling for a list of notations.

    Args:
        notations: the notations whose spelling needs to be fixed
        eraseFixedNotes: if True any previously fixed spelling is erased
        options: an EnharmonicOptions object. If not given a default is
            created. Many customizations can be modified here regarding the
            spelling algorithm
        spellingHistory: a SpellingHistory instance, if applicable

    Returns:
        nothing, spelling is fixed inplace

    These notations are preprocessed to signal the algorithm the measure
    boundaries. At these boundaries any fixed slots can be reset.

    Algorithm
    ~~~~~~~~~

    * We assume a chord for each notation
    * For each chord the highest already fixed pitch is picked, or the highest pitch
    * In a sliding window (the window size and hop are set in the options) the best
      spelling for the group is found. Fixed note classes are carried as fixed slots
      within a quarter-tone grid. For each slot in this grid an alteration direction
      is recorded. This determines if any further note in this slot should be spelled
      up (C#) or down (Db).
    * When the window slides to the right the first notes of the window might already
      be fixed
    """
    # First fix single notes and upper note of chords, then fix each chord
    notations = [n for n in notations
                 if not n.isRest and all(p > 10 for p in n.pitches)]

    if len(notations) == 1:
        n = notations[0]
        spellings = pitchSpellings(n)
        if options.debug:
            print(f"Single notation, {n}, fixed to {spellings}")
        for i, spelling in enumerate(spellings):
            n.fixNotename(spelling, i)
        return

    if spellingHistory is None:
        spellingHistory = SpellingHistory()

    if eraseFixedNotes:
        for n in notations:
            if n.fixedNotenames:
                n.fixedNotenames.clear()

    def anchorPitchIndex(n: Notation) -> int:
        if n.fixedNotenames:
            return max(n.fixedNotenames.keys())
        return len(n.pitches) - 1

    groupSize = min(options.groupSize, len(notations))
    for group in iterlib.window_fixed_size(notations, groupSize, options.groupStep):
        # Now we need the enharmonic variations but only from those
        # notes which are not fixed yet
        # Use either the highest fixed note or the highest note of each chord as reference
        # for each notation. Once there is a fixed reference for each notation the best
        # spelling for each chord is found, respecting those fixed points
        unfixedNotesIndexes = [i for i, n in enumerate(group) if not n.fixedNotenames]
        if not unfixedNotesIndexes:
            continue

        # These are the notenames within the window which are considered horizontally
        # Some might be fixed. Only one note per chord
        anchorIndexes = [anchorPitchIndex(n) for n in group]
        notenamesInGroup = [n.notename(idx) for idx, n in zip(anchorIndexes, group)]
        # notenamesInGroup = [n.notename(anchorPitchIndex(n)) for n in group]

        # The notes/chords which do not have any fixed notes
        unfixedNotes = [group[i] for i in unfixedNotesIndexes]

        # Since these notations do not have any fixed notes, take the highest
        # as the reference for horizontal fitting
        unfixedNotenames = [n.notename(len(n.pitches) - 1) for n in unfixedNotes]

        # Variations on those unset notenames. These are then replaced in the original group
        # and the whole group is weighted to find the best fit
        partialVariations = pt.enharmonic_variations(unfixedNotenames,
                                                     fixedslots=spellingHistory.slots)
        # If there are no variations (meaning that there are no solutions which respect
        # the fixed slots) we evaluate all solutions (force=True), forgetting about
        # any fixed slots.
        if not partialVariations:
            spellingHistory.clear()
            partialVariations = pt.enharmonic_variations(unfixedNotenames, force=True)

        # There should be at least one solution...
        assert partialVariations, f"{unfixedNotenames=}, {spellingHistory.slots=}"

        # Gather all variations to be analyzed later
        variations = []
        for variation in partialVariations:
            # We copy the original group and fill only the unset notes
            filledVar = notenamesInGroup.copy()
            for index, notename in zip(unfixedNotesIndexes, variation):
                filledVar[index] = notename
            variations.append(filledVar)
        validVariations = [v for v in variations if isEnharmonicVariantValid(v)]
        if not validVariations:
            validVariations = variations

        solution = _rateEnharmonicVariations(group, spellings=validVariations, anchors=anchorIndexes, options=options)

        if options.debug:
            print(f"DEBUG: Enharmonic spelling - orig. {notenamesInGroup}, "
                  f"horizontal solution={solution}")
        for idx, n in enumerate(group):
            if len(n) == 1:
                if not n.getFixedNotename():
                    n.fixNotename(solution[idx])
                    spellingHistory.addNotation(n)

            else:
                # A Chord
                n.fixNotename(solution[idx])
                fixedslots = spellingHistory.slots

                if nslots := n.fixedSlots():
                    if options.debug:
                        print(f"DEBUG: Enharmonic spelling - chord {n} - previous slots state:")
                        print(f"       {_reprslots(fixedslots, semitoneDivs=2)}")
                        print(f"       chords fixed slots: {_reprslots(fixedslots, semitoneDivs=2)}")

                    fixedslots = fixedslots.copy()
                    fixedslots.update(nslots)

                chordVariants = pt.enharmonic_variations(n.resolveNotenames(), fixedslots=fixedslots, force=True)
                chordVariants.sort(key=lambda variant: _rateChordSpelling(variant, options)[0])
                chordSolution = chordVariants[0]
                if options.debug:
                    print(f"DEBUG: Enharmonic spelling - chord {n} - solution: {chordSolution}")
                    print(f"       fixed slots: {_reprslots(fixedslots, semitoneDivs=2)}")
                    print(f"       variants: {chordVariants}")
                for i, notename in enumerate(chordSolution):
                    n.fixNotename(notename, idx=i)
                spellingHistory.addNotation(n)

    # Fix wrong accidentals
    # In pairs, we check glissandi and notes with inverted vertical position / pitch
    # (things like 4C# 4Db-)
    tiestart = None
    for n0, n1 in iterlib.window(notations, 2):
        if not n0.tiedPrev and n0.tiedNext:
            tiestart = n0
        elif n0.isRest or n0.isGracenote:
            tiestart = None
        if n0.isRest or n1.isRest or len(n0.pitches) > 1 or len(n1.pitches) > 1:
            continue
        if n0.pitches[0] < n1.pitches[0] and n0.verticalPosition() > n1.verticalPosition():
            # 4Db- : 4C#  -> 4C+ : 4Db
            # print(f"### 1", n0, n0.getAttachments('text'))

            n0.fixNotename(pt.enharmonic(n0.notename()))
            n1.fixNotename(pt.enharmonic(n1.notename()))
        elif n0.pitches[0] > n1.pitches[0] and n0.verticalPosition() < n1.verticalPosition():
            # 4C# : 4Db-  -> 4Db : 4C+
            # print(f"### 2", n0, n0.getAttachments('text'))

            n0.fixNotename(pt.enharmonic(n0.notename()))
            n1.fixNotename(pt.enharmonic(n1.notename()))
        elif (n0.gliss and
              not n0.tiedNext and
              n0.verticalPosition() == n1.verticalPosition()):
            n0fixed, n1fixed = _bestChordSpelling((n0.notename(), n1.notename()), options=options)
            n1.fixNotename(n1fixed)
            if n0.tiedPrev and tiestart is not None:
                tiestart.fixNotename(n0fixed)
            else:
                n0.fixNotename(n0fixed)

    return


def _verifyVariants(variants: list[tuple[str, ...]], slots):
    for variant in variants:
        for n in variant:
            notated = pt.notated_pitch(n)
            idx = notated.microtone_index(2)
            assert not slots.get(idx, 0) or slots[idx] == notated.alteration_direction(), \
                f"{variant=}, {idx=}, {notated.fullname=}, {slots=}"


