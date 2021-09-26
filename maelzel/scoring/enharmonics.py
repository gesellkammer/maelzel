from __future__ import annotations
from collections import defaultdict

import emlib.misc
from emlib import iterlib
from dataclasses import dataclass
import pitchtools as pt
import functools
from collections import deque
from .notation import Notation
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *


@dataclass
class EnharmonicOptions:
    groupSize: int = 5
    """Max size of a group to be evaluated for best enharmonic variant"""

    groupStep: int = 2

    fixedSlotMaxHistory: int = 4

    threeQuarterMicrotonePenalty:int = 10.01
    """penalty applied for 3/4 microtones like #+ or b- (prefer A+ over Bb-)"""

    respellingPenalty: int = 75
    """penalty for respelling a pitch (C+ / Db-)"""

    confusingIntervalPenalty: int = 21

    mispelledInterval: int = 50

    unisonAlterationPenalty: int = 12
    "penalty for C / C+"

    maxPenalty: int = 999999


_defaultEnharmonicOptions = EnharmonicOptions()


def isEnharmonicVariantValid(notes: List[str]) -> bool:
    pitches = [pt.n2m(n) for n in notes]
    for i0, i1 in iterlib.window(range(len(notes)), 2):
        p0 = pitches[i0]
        p1 = pitches[i1]
        n0 = notes[i0]
        n1 = notes[i1]
        if p0 < p1 and pt.vertical_position(n0) > pt.vertical_position(n1):
            return False
        if p0 > p1 and pt.vertical_position(n0) < pt.vertical_position(n1):
            return False
    return True


def groupPenalty(notes: List[str], options: EnharmonicOptions = None) -> Tuple[float, str]:
    if options is None:
        options = _defaultEnharmonicOptions
    total = 0
    penaltysources = []
    notated: List[pt.NotatedPitch] = [pt.notated_pitch(n) for n in notes]
    alterations = [n.alteration_direction() for n in notated
                   if n.is_black_key]
    numDirections = len(set(alterations))
    if numDirections > 1:
        total += 0

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
            penaltysources.append("threeQuarterMcrotone")
            total += options.threeQuarterMicrotonePenalty

    # Penalize successive microtones going in different directions
    # as the intervals are difficult to read
    for n0, n1 in iterlib.window(notated, 2):
        if n1.vertical_position - n0.vertical_position > 1 and n1.diatonic_alteration == -1.5:
            # C Eb-
            penaltysources.append("confusingInterval C Eb-")
            total += options.confusingIntervalPenalty * 3
        if n0.chromatic_alteration != 0 and n1.chromatic_alteration != 0:
            if(n0.alteration_direction(min_alteration=0.25) != n1.alteration_direction(0.25)):
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


def intervalsPenalty(notes: List[str], options: EnharmonicOptions
                     ) -> Tuple[float, str]:
    total = 0
    sources = []
    for n0, n1 in iterlib.window(notes, 2):
        penalty, source = intervalPenalty(n0, n1, options)
        total += penalty
        sources.append(source)
    sources = [s for s in sources if s]
    return total, ", ".join(sources)


def intervalPenalty(n0: str, n1: str, options: EnharmonicOptions
                    ) -> Tuple[float, str]:
    """
    Rate the penalty of the interval between n0 and n1

    Args:
        n0: the first notename
        n1: the second notename
        options: penalty options

    Returns:
        a tuple (penalty, penalty sources)
    """
    if pt.n2m(n0) > pt.n2m(n1):
        n0, n1 = n1, n0
    interval = pt.notated_interval(n0, n1)
    dpos, dpitch = interval
    assert dpitch >= 0
    if dpos < 0:
        # 4Eb- / 4D#
        return options.maxPenalty, "Really confusing interval"

    if dpitch == 0:
        if dpos == 0:
            # same pitch, no penalty
            return 0, ""
        return options.maxPenalty, "Respelled pitch"

    if (dpos > 0 and dpitch < 0) or (dpos < 0 and dpitch > 0):
        # ex: 4C# 4Db- or 4E+ 4Fb
        return options.maxPenalty, "Inverse pitch"

    dpos7 = dpos % 7
    dpitch12 = dpitch % 12
    notated0 = pt.notated_pitch(n0)
    notated1 = pt.notated_pitch(n1)
    penalties = []
    _ = penalties.append
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
            _((options.unisonAlterationPenalty, f"unison"))
        else:
            _((options.unisonAlterationPenalty, "unisonAlteration"))
    elif dpos == 1:
        # 4C 4D#, 4G 4A#  -> better 4C 4Eb or 4G 4Bb
        if dpitch >= 3:
            _((options.mispelledInterval, "mispelled"))
    elif dpos7 == 2:
        if dpitch12 > 5:
            _((options.mispelledInterval , f"mispelledInterval"))
        elif dpitch12 >= 4.5:
            _((options.confusingIntervalPenalty, "Db/F+"))
        elif dpitch12 <= 1.5:
            # 4F# 4Ab- (dpitch=2)
            _((options.mispelledInterval*2, f"mispelledInterval!"))
        elif dpitch12 <= 2:
            _((options.mispelledInterval, f"mispelledInterval"))
    elif dpos7 == 3:
        # 4D 4Gb (dpitch=4)
        if dpitch12 < 3.5:
            # 4D# 4Gb- (dpitch=3)
            _((options.confusingIntervalPenalty*2, "confusing interval"))
        elif dpitch12 == 3.5 and abs(notated0.diatonic_alteration) == 1.5 or abs(notated1.diatonic_alteration) == 1.5:
            # 4D 4Gb-
            _((options.confusingIntervalPenalty * 10, "confusing"))
        elif dpitch12 <= 4:
            _((options.mispelledInterval, "mispelled"))
        elif dpitch12 == 4.5:
            _((options.confusingIntervalPenalty*1.5, "D/G-"))
        elif dpitch12 >= 7:
            # 4Db 4G#
            _((options.mispelledInterval, "mispelled"))
    elif dpos7 == 4: # a 5th
        if dpitch12 >= 9:
            # 4Db 4A#
            _((options.mispelledInterval, "mispelled"))
        elif dpitch12 <= 5:
            # 4D# 4Ab
            _((options.mispelledInterval, "mispelled"))
    elif dpos7 == 5: # a 6th
        # C# Ab (7) / Db B (10)
        if not 7 < dpitch12 < 10:
            _((options.mispelledInterval, "mispelled"))
    total = sum(p for p, s in penalties)
    source = ", ".join(s for p, s in penalties) + f"({n0}/{n1})"
    return total, source


def _enharmonicPenalty(notes: List[str], options:EnharmonicOptions) -> float:
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
    total = intervalpenalty + grouppenalty
    # print(notes, total, "intervals", intervalpenalty, sources0, "group", grouppenalty, sources1)
    return total


class _SpellingHistory:
    def __init__(self, itemHistory=4, pitchHistory=5, divsPerSemitone=2):
        self.itemHistory = itemHistory
        self.pitchHistory = pitchHistory
        self.dequelen = itemHistory
        self.deque: deque[List[pt.NotatedPitch]] = deque(maxlen=self.dequelen)
        self.divsPerSemitone= divsPerSemitone
        numslots = 12 * divsPerSemitone
        self.slots = {idx: 0 for idx in range(numslots)}
        self.refcount = {idx: 0 for idx in range(numslots)}

    def clear(self):
        self.deque.clear()
        numslots = 12 * self.divsPerSemitone
        self.slots = {idx:0 for idx in range(numslots)}
        self.refcount = {idx:0 for idx in range(numslots)}

    def add(self, items: List[str]) -> None:
        ns = [pt.notated_pitch(item) for item in items]
        numpitches = sum(len(item) for item in self.deque)
        if len(self.deque) == self.itemHistory or numpitches>=self.pitchHistory:
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
                raise ValueError(f"spelling error with {item}, spelling already fixed "
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
        idx2name = lambda idx: pt.m2n(60+idx/2)[1:]
        fixed = {f"{idx}:{idx2name(idx)}": val for idx, val in self.slots.items()
                 if val != 0}
        print(fixed)

    def addNotation(self, notation:Notation):
        notenames = notation.notenames
        for notename in notenames:
            if not self.spellingOk(notename):
                self.dump()
                raise ValueError(f"Spelling for {notename} already set "
                                 f"(notation={notenames})")
        self.add(notenames)


def _rateChordSpelling(notes: List[str], options: EnharmonicOptions) -> Tuple[float, str]:
    totalpenalty = 0.
    sources = []
    for a, b in iterlib.combinations(notes, 2):
        penalty, source = intervalPenalty(a, b, options)
        totalpenalty += penalty
        sources.append(source)
    sourcestr = ":".join(sources)
    # print(notes, totalpenalty, sourcestr)
    return totalpenalty, sourcestr


def fixEnharmonicsInPlace(notations: List[Notation], eraseFixedNotes=True,
                          options: EnharmonicOptions = None,
                          ) -> None:
    # First fix single notes and upper note of chords, then fix each chord
    notations = [n for n in notations if not n.isRest]

    if options is None:
        options = _defaultEnharmonicOptions

    spellingHistory = _SpellingHistory()

    if eraseFixedNotes:
        for n in notations:
            if n.fixedNotenames:
                n.fixedNotenames.clear()

    groupSize = min(options.groupSize, len(notations))
    for group in iterlib.window_fixed_size(notations, groupSize, options.groupStep):
        # Now we need the enharmonic variations but only from those
        # notes which are not fixed yet
        notenamesInGroup = [n.notename() for n in group]
        unfixed = [n for n in group if n.getFixedNotename() is None]
        if not unfixed:
            continue
        unfixedNotenames = [n.notename() for n in unfixed]
        unfixedIndexes = [i for i, n in enumerate(group) if n.getFixedNotename() is None]
        assert unfixedIndexes
        partialVariations = pt.enharmonic_variations(unfixedNotenames,
                                                     fixedslots=spellingHistory.slots)
        if not partialVariations:
            spellingHistory.clear()
            partialVariations = pt.enharmonic_variations(unfixedNotenames)

        # assert partialVariations, f"{unfixedNotenames=}, {spellingHistory.slots=}"
        variations = []
        for variation in partialVariations:
            filledVar = notenamesInGroup.copy()
            for index, notename in zip(unfixedIndexes, variation):
                filledVar[index] = notename
            variations.append(filledVar)
        validVariations = [v for v in variations if isEnharmonicVariantValid(v)]
        if not validVariations:
            validVariations = variations
        validVariations.sort(key=functools.partial(_enharmonicPenalty, options=options))
        solution = validVariations[0]
        for idx, n in enumerate(group):
            if len(n) == 1:
                if not n.getFixedNotename():
                    n.fixNotename(solution[idx])
                    spellingHistory.addNotation(n)

        # for idx, n in enumerate(group):
            elif len(n) > 1:
                n.fixNotename(solution[idx], 0)
                fixedslots = spellingHistory.slots.copy()
                fixedslots[n.pitchIndex(2, 0)] = n.accidentalDirection(0)
                assert all(abs(value) <= 1 for value in fixedslots.values())
                chordVariants = pt.enharmonic_variations(n.notenames, fixedslots=fixedslots)
                _verifyVariants(chordVariants, fixedslots)
                chordVariants.sort(key=lambda variant:_rateChordSpelling(variant, options)[0])
                chordSolution = chordVariants[0]
                for i, notename in enumerate(chordSolution):
                    n.fixNotename(notename, idx=i)
                spellingHistory.addNotation(n)

    # Fix wrong accidentals
    # In pairs, we check glissandi and notes with inverted vertical position / pitch
    # (things like 4C# 4Db-)
    for n0, n1 in iterlib.window(notations, 2):
        if n0.isRest or n1.isRest:
            continue
        if n0.pitches[0]<n1.pitches[0] and \
                n0.verticalPosition()>n1.verticalPosition():
            # 4Db- : 4C#  -> 4C+ : 4Db
            n0.fixNotename(pt.enharmonic(n0.notename()))
            n1.fixNotename(pt.enharmonic(n1.notename()))
        elif n0.pitches[0]>n1.pitches[0] and \
                n0.verticalPosition()<n1.verticalPosition():
            # 4C# : 4Db-  -> 4Db : 4C+
            n0.fixNotename(pt.enharmonic(n0.notename()))
            n1.fixNotename(pt.enharmonic(n1.notename()))

    return

def _verifyVariants(variants: List[Tuple[str, ...]], slots):
    for variant in variants:
        for n in variant:
            notated = pt.notated_pitch(n)
            idx = notated.microtone_index(2)
            assert not slots.get(idx, 0) or slots[idx] == notated.alteration_direction(), \
                f"{variant=}, {idx=}, {notated.fullname=}, {slots=}"

