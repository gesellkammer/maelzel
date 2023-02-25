from __future__ import annotations
from maelzel.core.event import MEvent
from maelzel.core import _util
from maelzel.core._common import *
from maelzel.common import F, asF
from emlib import iterlib
from emlib import misc
from maelzel.core.synthevent import PlayArgs, SynthEvent
from maelzel.core.workspace import getConfig
from maelzel import scoring


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    from maelzel.core._typedefs import *
    from maelzel.scorestruct import ScoreStruct
    from maelzel.core.workspace import Workspace
    from maelzel.core.config import CoreConfig
    from maelzel.core.event import Note


class Line(MEvent):
    """
    A Line is a sequence of breakpoints

    Args:
        bps: breakpoints, a tuple/list of the form (delay, pitch, [amp=1, ...]), where
            delay is the time offset to the beginning of the line; pitch is the pitch
            as notename or midinote and amp is an amplitude between 0-1. If values are
            missing from one row they are carried from the previous
        start: time offset of the line itself
        label: a label to add to the line
        relative: if True, the first value of each breakpoint is a time offset
            from previous breakpoint

    A bp has the form ``(delay, pitch, [amp=1, ...])``, where:

    - delay is the time offset to the first breakpoint.
    - pitch is the pitch as midinote or notename
    - amp is the amplitude (0-1), optional

    pitch, amp and any other following data can be 'carried'::

        Line((0, "D4"), (1, "D5", 0.5), ..., fade=0.5)

    Also possible::

    >>> bps = [(0, "D4"), (1, "D5"), ...]
    >>> Line(bps)   # without *

    a Line stores its breakpoints as: ``[delayFromFirstBreakpoint, pitch, amp, ...]``

    Attributes:
        bps: the breakpoints of this line, a list of tuples of the form
            ``(delay, pitch, [amp, ...])``, where delay is always relative
            to the start of the line (the delay of the first breakpoint is always 0)
    """

    __slots__ = ('bps', 'dynamic', 'tied', 'gliss')
    _acceptsNoteAttachedSymbols = True

    def __init__(self,
                 *bps,
                 start: num_t = None,
                 label="",
                 relative=False,
                 dynamic='',
                 tied=False,
                 gliss=False):
        # [[[0, 60, 1], [1, 60, 2]]]
        if len(bps) == 1 and isinstance(bps[0], list) and isinstance(bps[0][0], (tuple, list)):
            bps = bps[0]

        if any(len(bp) < 2 for bp in bps):
            raise ValueError("A breakpoint should be at least (delay, pitch)", bps)

        if len(bps[0]) == 2:
            bps[0].append(1.)

        l = len(bps)
        if any(len(bp) != l for bp in bps):
            bps = _util.carryColumns(bps)

        bps = _util.as2dlist(bps)

        for bp in bps:
            bp[0] = asF(bp[0])
            bp[1] = _util.asmidi(bp[1])

        if relative:
            now = 0
            for bp in bps:
                now += bp[0]
                bp[0] = now

        if bps[0][0] > 0 and start is not None:
            dt = bps[0][0]
            start += dt
            for row in bps:
                row[0] -= dt

        assert all(bp1[0] > bp0[0] for bp0, bp1 in iterlib.pairwise(bps))

        super().__init__(dur=bps[-1][0], offset=start, label=label)
        self.bps: list[list] = bps
        """The breakpoints of this line, a list of tuples (delay, pitch, [amp, ...])"""

        self.dynamic = dynamic
        self.gliss = gliss
        self.tied = tied

    def resolveOffset(self) -> F:
        if self.offset is not None:
            return self.offset
        return self.bps[0][0]

    def resolveDur(self, start: time_t = None) -> F:
        return self.bps[-1][0] - self.bps[0][0]

    def offsets(self) -> list[F]:
        """ Return absolute offsets of each breakpoint """
        start = self.offset or F(0)
        return [bp[0] + start for bp in self.bps]

    def translateBreakpointsToAbsTime(self,
                                      score: ScoreStruct,
                                      asfloat=True
                                      ) -> list[list[num_t, ...]]:
        """
        Translate beat to absolute time within the breakpoints of this Line

        Args:
            score: the scorestructure to use to translate quarter notes to
                abs time
            asfloat: if True, convert all times to float

        Returns:
            a copy of this Lines breakpoints where all timing is given in
            absolute time
        """
        start = self.offset or F(0)
        bps = []
        for bp in self.bps:
            bp2 = bp.copy()
            t = score.beatToTime(bp[0] + start)
            if asfloat:
                t = float(t)
            bp2[0] = t
            bps.append(bp2)
        return bps

    def _synthEvents(self, playargs: PlayArgs, workspace: Workspace
                     ) -> list[SynthEvent]:
        conf = workspace.config
        if self.playargs:
            playargs.overwriteWith(self.playargs)
        playargs.fillDefaults(conf)
        bps = self.translateBreakpointsToAbsTime(workspace.scorestruct, asfloat=True)
        return [SynthEvent.fromPlayArgs(bps, playargs=playargs)]

    def __hash__(self):
        rowhashes = [hash(tuple(bp)) for bp in self.bps]
        attrs = (self.offset, self.dynamic, self.gliss, self.tied)
        rowhashes.extend(attrs)
        return hash(tuple(rowhashes))

    def __repr__(self):
        return f"Line(start={self.offset}, bps={self.bps})"

    def quantizePitch(self, step=0) -> Line:
        """
        Returns a new line, rounded to step

        Some breakpoints might not be relevant after quantization. These
        will be simplified, meaning that the returned line might have less
        breakpoints than the original line.

        Args:
            step: the semitone division used (0.5 = quantize to nearest 1/4 tone)

        Returns:
            a line with its pitches quantized by the given step
        """
        if step == 0:
            step = 1 / getConfig()['semitoneDivisions']
        bps = [(bp[0], _util.quantizeMidi(bp[1], step)) + bp[2:]
               for bp in self.bps]
        if len(bps) >= 3:
            bps = misc.simplify_breakpoints(bps, coordsfunc=lambda bp: (bp[0], bp[1]),
                                            tolerance=0.01)
        return Line(bps)

    def scoringEvents(self,
                      groupid: str = None,
                      config: CoreConfig = None
                      ) -> list[scoring.Notation]:
        if config is None:
            config = getConfig()
        start = self.offset or F(0)
        # groupid = scoring.makeGroupId(groupid)
        notations: list[scoring.Notation] = []

        for bp0, bp1 in iterlib.pairwise(self.bps):
            pitch = bp0[1]
            dur = bp1[0] - bp0[0]
            offset = bp0[0] + start
            ev = scoring.makeNote(pitch=pitch, offset=offset, duration=dur,
                                  gliss=bp0[1] != bp1[1], group=groupid)
            if bp0[1] == bp1[1]:
                ev.tiedNext = True
            notations.append(ev)

        if self.bps[-1][1] != self.bps[-2][1]:
            # add a last note if last pair needed a gliss (to have a destination note)
            n = notations[-1]
            n.gliss = True
            lastbp = self.bps[-1]
            notations.append(scoring.makeNote(pitch=lastbp[1],
                                              offset=lastbp[0] + start,
                                              gracenote=True,
                                              group=groupid))
        if notations:
            scoring.removeOverlap(notations)
            annot = self._scoringAnnotation()
            if annot:
                notations[0].addText(annot)
        if self.symbols:
            for symbol in self.symbols:
                symbol.applyToTiedGroup(notations)
        if self.tied:
            logger.warning("Tied lines are not yet supported")
        if self.gliss:
            logger.warning("Gliss at the end of lines is not yet supported")

        if self.dynamic:
            notations[0].dynamic = self.dynamic
        return notations

    def timeTransform(self, timemap: Callable[[num_t], num_t], inplace=False) -> Line:
        if not inplace:
            bps = []
            for bp in self.bps:
                t1 = timemap(bp[0] + self.offset)
                bp2 = bp.copy()
                bp2[0] = t1
                bps.append(bp2)
            return Line(bps, label=self.label)
        else:
            for bp in self.bps:
                bp[0] = timemap(bp[0] + self.offset)

    def dump(self, indents=0):
        attrs = []
        if self.offset:
            attrs.append(f"start={self.offset}")
        if self.label:
            attrs.append(f"label={self.label}")
        infostr = ", ".join(attrs)
        print("Line:", infostr)
        rows = []
        for bp in self.bps:
            row = ["%.6g" % _ for _ in bp]
            rows.append(row)
        headers = ("start", "pitch", "amp", "p4", "p5", "p6", "p7", "p8")
        misc.print_table(rows, headers=headers, showindex=False, )

    def timeShift(self, timeoffset: time_t) -> Line:
        return Line(self.bps, start=(self.offset or F(0)) + timeoffset)

    def pitchTransform(self, pitchmap: Callable[[float], float]) -> Line:
        newpitches = [pitchmap(bp[1]) for bp in self.bps]
        newbps = self.bps.copy()
        for bp, pitch in zip(newbps, newpitches):
            bp[1] = pitch
        return self.clone(bps=newbps)


def makeLine(notes: list[Note], workspace: Workspace = None) -> Line:
    assert all(n0.end == n1.offset for n0, n1 in iterlib.pairwise(notes))
    bps = []
    for note in notes:
        bp = [note.offset, note.pitch, note.resolveAmp(workspace=workspace)]
        bps.append(bp)
    lastnote = notes[-1]
    if lastnote.dur > 0:
        pitch = lastnote.gliss if lastnote.gliss else lastnote.pitch
        bps.append([lastnote.end, pitch, lastnote.resolveAmp()])
    return Line(bps, label=notes[0].label)
