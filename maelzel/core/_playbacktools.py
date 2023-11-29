from __future__ import annotations
from .synthevent import SynthEvent
from typing import Sequence
import csoundengine
import maelzel.core as mc
from datetime import datetime
import os
import math


def collectEvents(events,
                  eventparams: dict,
                  workspace: mc.workspace.Workspace
                  ) -> tuple[list[SynthEvent], list[csoundengine.event.Event]]:
    synthevents = []
    sessionevents = []
    for ev in events:
        if isinstance(ev, (list, tuple)):
            if isinstance(ev[0], SynthEvent):
                synthevents.extend(ev)
            else:
                evs, sessionevs = collectEvents(ev, eventparams=eventparams, workspace=workspace)
                synthevents.extend(evs)
                sessionevents.extend(sessionevs)
        elif isinstance(ev, SynthEvent):
            synthevents.append(ev)
        elif isinstance(ev, csoundengine.event.Event):
            sessionevents.append(ev)
        else:
            synthevents.extend(ev.events(workspace=workspace, **eventparams))
    return synthevents, sessionevents


def makeRecordingFilename(ext=".wav", prefix="rec-"):
    """
    Generate a new filename for a recording.

    This is used when rendering and no outfile is given

    Args:
        ext: the extension of the soundfile (should start with ".")
        prefix: a prefix used to identify this recording

    Returns:
        an absolute path. It is guaranteed that the filename does not exist.
        The file will be created inside the recording path
        (see :meth:`Workspace.recordPath() <maelzel.core.workspace.Workspace.recordPath>`)
    """
    path = mc.getWorkspace().recordPath()
    assert ext.startswith(".")
    base = datetime.now().isoformat(timespec='milliseconds')
    if prefix:
        base = prefix + base
    out = os.path.join(path, base + ext)
    assert not os.path.exists(out)
    return out


def nchnlsForEvents(events: list[SynthEvent]) -> int:
    """
    Analyze the events and determine the number of channels needed

    Args:
        events: the events to analyze

    Returns:
        the number of channels needed to render these events

    """
    return max(int(math.ceil(ev.resolvedPosition() + ev.chan)) for ev in events)

