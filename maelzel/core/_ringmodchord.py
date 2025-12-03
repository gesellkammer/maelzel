from __future__ import annotations
from maelzel.core.event import Chord, Note
from maelzel.core.synthevent import PlayArgs, SynthEvent
from maelzel.core.workspace import Workspace
from maelzel.common import F
from maelzel.music import combtones
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
        struct = workspace.scorestruct
        startbeat = parentOffset + self.relOffset()
        startsecs = float(struct.beatToTime(startbeat))
        endsecs = float(struct.beatToTime(startbeat + self.dur))
        amps = self.resolveAmps(config=conf, dyncurve=workspace.dynamicCurve)
        rmpairs1 = combtones.ringmodWithAmps(self.pitches, amps, merge='sum')
        if not self.gliss:
            rmpairs2 = rmpairs1
        else:
            endpitches = self.pitches if not self.gliss else self.glissTargetPitches()
            rmpairs2 = combtones.ringmodWithAmps(endpitches, amps, merge='sum')
            assert len(rmpairs1) == len(rmpairs2)
        if conf['chordAdjustGain']:
            gain = playargs.get('gain', 1.0)
            playargs['gain'] = gain / math.sqrt(len(rmpairs1))
        sustain = playargs.get('sustain', 0.)
        transp = playargs.get('transpose', 0.)
        synthevents = []
        for (pitch1, amp1), (pitch2, amp2) in zip(rmpairs1, rmpairs2):
            bps = [(startsecs, pitch1 + transp, amp1),
                   (endsecs, pitch2 + transp, amp2)]
            if sustain:
                bps.append((endsecs + sustain, pitch2 + transp, amp2))
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
        amps = self.resolveAmps(config=workspace.config, dyncurve=workspace.dynamicCurve)
        pairs = combtones.ringmodWithAmps(self.pitches, amps, merge='sum')
        notes = [Note(pitch, amp=amp) for pitch, amp in pairs]
        if not self.gliss:
            return Chord(notes=notes, dur=self.dur)
        gliss = self.glissTargetPitches()
        glisspitches = combtones.ringmod(gliss)
        return Chord(notes, dur=self.dur, gliss=glisspitches)

