from __future__ import annotations
from maelzel.core.event import Chord, Note
from maelzel.core.synthevent import PlayArgs, SynthEvent
from maelzel.core.workspace import Workspace
from maelzel.common import F, asF
from maelzel.music import combtones
from maelzel.core.config import CoreConfig
from maelzel import scoring
from maelzel.core import symbols
import math


class RingmodChord(Chord):

    def _synthEvents(self,
                     playargs: PlayArgs,
                     parentOffset: F,
                     workspace: Workspace
                     ) -> list[SynthEvent]:
        conf = workspace.config
        if self.playargs:
            playargs = playargs.updated(self.playargs)
        startsecs, endsecs = self.timeRangeSecs(parentOffset=parentOffset, scorestruct=workspace.scorestruct)
        amps = self.resolveAmps(config=conf, dyncurve=workspace.dynamicCurve)
        endpitches = self.pitches if not self.gliss else self.resolveGliss()
        rmpairs1 = combtones.ringmodWithAmps(self.pitches, amps, unique=False)
        rmpairs2 = combtones.ringmodWithAmps(endpitches, amps, unique=False)
        assert len(rmpairs1) == len(rmpairs2)
        if conf['chordAdjustGain']:
            gain = playargs.get('gain', 1.0)
            playargs['gain'] = gain / math.sqrt(len(rmpairs1))
        sustain = playargs.get('sustain', 0.)
        transp = playargs.get('transpose', 0.)
        synthevents = []
        for (pitch1, amp1), (pitch2, amp2) in zip(rmpairs1, rmpairs2):
            bps = [[startsecs, pitch1 + transp, amp1],
                   [endsecs, pitch2 + transp, amp2]]
            if sustain:
                bps.append([endsecs + sustain, pitch2 + transp, amp2])
            synthevents.append(SynthEvent.fromPlayArgs(bps=bps, playargs=playargs))
        return synthevents

    def bands(self) -> Chord:
        """
        Return the resulting bands (combination tones of each pair) as a Chord

        Returns:
            a Chord containing the ringmod bands resulting from calculating
            the summation and difference tones of each pair within this chord
        """
        workspace = Workspace.active
        conf = workspace.config
        amps = self.resolveAmps(config=conf, dyncurve=workspace.dynamicCurve)
        gliss = False if not self.gliss else self.resolveGliss()
        pairs = combtones.ringmodWithAmps(self.pitches, amps, unique=False)
        if gliss:
            glissbands = combtones.ringmod(gliss)
            notes = [Note(pitch, amp=amp, gliss=glisstarget)
                     for (pitch, amp), glisstarget in zip(pairs, glissbands)]
        else:
            notes = [Note(pitch, amp=amp) for pitch, amp in pairs]
        return self.clone(notes=notes)

    def scoringEvents(self,
                      groupid='',
                      config: CoreConfig = None,
                      parentOffset: F | None = None
                      ) -> list[scoring.Notation]:
        notes = self.notes.copy()
        bandnotes = self.bands().notes
        for note in bandnotes:
            note.addSymbol(symbols.Notehead('harmonic', size=0.85))
        notes.extend(bandnotes)
        if parentOffset is None:
            parentOffset = self.parent.absOffset() if self.parent else F(0)
        if self.gliss:
            gliss = self.resolveGliss()
            glisschord = RingmodChord(gliss, dur=0, offset=self.relOffset() + self.dur)
            events = self.clone(notes=notes, gliss=True).scoringEvents(groupid=groupid, config=config, parentOffset=parentOffset)
            events.extend(glisschord.scoringEvents(groupid=groupid, config=config, parentOffset=parentOffset))
            return events
        else:
            return self.clone(notes=notes).scoringEvents(groupid=groupid, config=config, parentOffset=parentOffset)








