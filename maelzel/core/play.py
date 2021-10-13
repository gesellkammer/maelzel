"""
This module handles playing of events

Each Note, Chord, Line, etc, can express its playback in terms of CsoundEvents

A CsoundEvent is a score line with a number of fixed fields,
user-defined fields and a sequence of breakpoints

A breakpoint is a tuple of values of the form (offset, pitch [, amp, ...])
The size if each breakpoint and the number of breakpoints are given
by inumbps, ibplen

An instrument to handle playback should be defined with `defPreset` which handles
breakpoints and only needs the audio generating part of the csound code.

Whenever a note actually is played with a given preset, this preset is
 sent to the csound engine and instantiated/evaluated.

Examples
~~~~~~~~

.. code::

    from maelzel.core import *
    f0 = n2f("1E")
    notes = [Note(f2m(i*f0), dur=0.5) for i in range(20)]
    play.defPreset("detuned", r'''

    ''')

"""
from __future__ import annotations
import os

from datetime import datetime

import csoundengine

from .config import logger
from .workspace import activeConfig, activeWorkspace, recordPath
from . import tools
from .presetbase import *
from .presetman import presetManager, csoundPrelude as _prelude
from .errors import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    from .csoundevent import CsoundEvent
    from .musicobjbase import MusicObj

__all__ = ('OfflineRenderer',
           'playEvents',
           'recEvents')

class PlayEngineNotStarted(Exception): pass


_invalidVariables = {"kfreq", "kamp", "kpitch"}


class OfflineRenderer:
    def __init__(self, outfile:str=None, sr=None, ksmps=64, quiet:bool=None):
        w = activeWorkspace()
        cfg = activeConfig()
        self.outfile = outfile
        self.a4 = w.a4
        self.sr = sr or cfg['rec.sr']
        self.quiet = quiet
        self.ksmps = ksmps
        self.renderer = presetManager.makeRenderer(sr, ksmps=ksmps)
        self.events: List[CsoundEvent] = []

    def registerInstr(self, instr: csoundengine.Instr) -> None:
        self.renderer.registerInstr(instr)

    def play(self, obj: MusicObj, **kws) -> List[csoundengine.offline.ScoreEvent]:
        events = obj.events(**kws)
        scoreEvents = [self.schedEvent(ev) for ev in events]
        return scoreEvents

    def schedEvent(self, event: CsoundEvent) -> csoundengine.offline.ScoreEvent:
        """
        Schedule a CsoundEvent as returned by MusicObj.events()

        Args:
            event: a CsoundEvent, as returned

        Returns:
            a ScoreEvent

        See Also:
            sched
        """
        return _schedCsoundEvent(self.renderer, event)

    def definedInstrs(self) -> Dict[str, csoundengine.Instr]:
        """
        Get all instruments available within this OfflineRenderer

        All presets and all extra intruments registered at the active
        Session (as returned via getPlaySession) are available

        Returns:
            dict {instrname: csoundengine.Instr} with all instruments available

        """
        instrs = {}
        instrs.update(self.renderer.registeredInstrs())
        instrs.update(getPlaySession().registeredInstrs())
        return instrs

    def sched(self,
              instrname: str,
              delay=0.,
              dur=-1.,
              priority=1,
              pargs: Union[List[float], Dict[str, float]] = None,
              tabargs: Dict[str, float] = None,
              **kws) -> csoundengine.offline.ScoreEvent:
        """
        Schedule a csound event

        This method should be used to schedule non-preset based instruments
        when rendering offline (things like global effects, for example),
        similarly to how a user might schedule a non-preset based instrument
        in real-time.

        Args:
            instrname: the instr. name
            delay: start time
            dur: duration
            priority: priority of the event
            pargs: any pargs passed to the instr., starting at p5
            tabargs: table args accepted by the instr.
            **kws: named pargs

        Returns:
            the offline.ScoreEvent, which can be used as a reference by other
            offline events

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> scale = Chain([Note(n) for n in "4C 4D 4E 4F 4G".split()])
            >>> play.getPlaySession().defInstr('reverb', r'''
            ... |kfeedback=0.6|
            ... amon1, amon2 monitor
            ... a1, a2 reverbsc amon1, amon2, kfeedback, 12000, sr, 0.6
            ... outch 1, a1-amon1, 2, a2-amon2
            ... ''')
            >>> offlinerenderer = play.OfflineRenderer()

        """
        if not self.renderer.isInstrDefined(instrname):
            session = getPlaySession()
            instr = session.getInstr(instrname)
            if not instr:
                logger.error(f"Unknown instrument {instrname}. "
                             f"Defined instruments: {self.renderer.registeredInstrs().keys()}")
                raise ValueError(f"Instrument {instrname} unknown")
            self.renderer.registerInstr(instr)
        return self.renderer.sched(instrname=instrname, delay=delay, dur=dur,
                                   priority=priority, pargs=pargs,
                                   tabargs=tabargs,
                                   **kws)


    def render(self, outfile:str=None, wait=None, quiet=None, openWhenDone=False
               ) -> str:
        """
        Render the events scheduled until now.

        Args:
            outfile: the soundfile to generate. Use "?" to save via a GUI dialog,
                None will render to a temporary file
            wait: if True, wait until rendering is done
            quiet: if True, supress all output generated by csound itself
                (print statements and similar opcodes still produce output)
            openWhenDone: if True, open the rendered soundfile in the default
                application

        Returns:
            the path of the renderer file
        """
        cfg = activeConfig()
        if outfile is None:
            outfile = self.outfile
        if outfile == '?':
            outfile = tools.saveRecordingDialog()
            if not outfile:
                raise CancelledError("Render operation was cancelled")
        elif not outfile:
            outfile = _makeRecordingFilename(ext=".wav")
        outfile = tools.normalizeFilename(outfile)
        if quiet is None:
            quiet = self.quiet if self.quiet is not None else cfg['rec.quiet']
        self.renderer.render(outfile=outfile, wait=wait, quiet=quiet,
                             openWhenDone=openWhenDone)
        return outfile

    def getCsd(self) -> str:
        """
        Return the .csd as string
        """
        return self.renderer.generateCsd()

    def writeCsd(self, outfile:str='?') -> str:
        """
        Write the .csd which would render all events scheduled until now

        Args:
            outfile: the path of the saved .csd

        Returns:
            the outfile
        """
        csdstr = self.getCsd()
        if outfile == "?":
            outfile = tools.selectFileForSave("saveCsdLastDir", filter="Csd (*.csd)")
            if not outfile:
                raise CancelledError("Save operation cancelled")
        with open(outfile, "w") as f:
            f.write(csdstr)
        return outfile

    def __enter__(self):
        workspace = activeWorkspace()
        self._oldRenderer = workspace.renderer
        workspace.renderer = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            # There was an exception since entering
            logger.warning("Offline rendering aborted")
            return
        w = activeWorkspace()
        w.renderer = self._oldRenderer
        if self.outfile is None:
            self.outfile = _makeRecordingFilename()
            logger.info(f"Rendering to {self.outfile}")
        self.render(outfile=self.outfile, wait=True)


def _schedCsoundEvent(renderer: csoundengine.Renderer, event: CsoundEvent,
                      instrIndex: Dict[str, csoundengine.Instr] = None
                      ) -> csoundengine.offline.ScoreEvent:
    instrdef = renderer.getInstr(event.instr)
    if instrdef is None:
        instrdef = instrIndex.get(event.instr) if instrIndex else None
        if instrdef is None:
            raise KeyError(f"instr {event.instr} not defined")
        renderer.registerInstr(instrdef)
    args = event.getPfields(numchans=instrdef.numchans)
    return renderer.sched(event.instr, delay=event.delay, dur=event.dur,
                          pargs=args[3:], priority=event.priority,
                          tabargs=event.namedArgs)


def recEvents(events: List[CsoundEvent], outfile:str=None,
              sr:int=None, wait:bool=None, ksmps:int=None,
              quiet=None
              ) -> str:
    """
    Record the events to a soundfile

    Args:
        events: a list of events as returned by .events(...)
        outfile: the generated file. If left unset, a file inside the recording
            path is created (see `recordPath`). Use "?" to save via a GUI dialog
        sr: sample rate of the soundfile
        ksmps: number of samples per cycle (config 'rec.ksmps')
        wait: if True, wait until recording is finished. If None,
            use the config 'rec.block'
        quiet: if True, supress debug information when calling
            the csound subprocess

    Returns:
        the path of the generated soundfile

    Example::

        a = Chord("A4 C5", start=1, dur=2)
        b = Note("G#4", dur=4)
        events = sum([
            a.events(chan=1),
            b.events(chan=2, gain=0.2)
        ], [])
        recEvents(events, outfile="out.wav")

    See Also
    ~~~~~~~~

    :class:`OfflineRenderer`
    """
    offlineRenderer = OfflineRenderer(sr=sr, ksmps=ksmps)
    for ev in events:
        offlineRenderer.schedEvent(ev)
    offlineRenderer.render(outfile=outfile, wait=wait, quiet=quiet)
    return outfile


def _path2name(path):
    return os.path.splitext(os.path.split(path)[1])[0].replace("-", "_")


def _makeRecordingFilename(ext=".wav", prefix="rec-"):
    """
    Generate a new filename for a recording.

    This is used when rendering and no outfile is given

    Args:
        ext: the extension of the soundfile (should start with ".")
        prefix: a prefix used to identify this recording

    Returns:
        an absolute path. It is guaranteed that the filename does not exist.
        The file will be created inside the recording path (see ``state.recordPath``)
    """
    path = recordPath()
    assert ext.startswith(".")
    base = datetime.now().isoformat(timespec='milliseconds')
    if prefix:
        base = prefix + base
    out = os.path.join(path, base + ext)
    assert not os.path.exists(out)
    return out


def _registerPresetInSession(preset: PresetDef,
                             session:csoundengine.session.Session
                             ) -> csoundengine.Instr:
    """
    Create and register a :class:`csoundengine.instr.Instr` from a preset

    Args:
        preset: the PresetDef.
        session: the session to manage the instr

    Returns:
        the registered Instr
    """
    # each preset caches the generated instr
    instr = preset.makeInstr()
    # registerInstr checks itself if the instr is already defined
    session.registerInstr(instr)
    return instr


def _soundfontToTabname(sfpath: str) -> str:
    path = os.path.abspath(sfpath)
    return f"gi_sf2func_{hash(path)%100000}"


def _soundfontToChannel(sfpath:str) -> str:
    basename = os.path.split(sfpath)[1]
    return f"_sf:{basename}"


def startPlayEngine(numChannels=None, backend=None) -> csoundengine.Engine:
    """
    Start the play engine

    If an engine is already active, nothing happens, even if the
    configuration is different. To start the play engine with a different
    configuration, stop the engine first.

    Args:
        numChannels: the number of output channels, overrides config 'play.numChannels'
        backend: the audio backend used, overrides config 'play.backend'
    """
    config = activeConfig()
    engineName = config['play.engineName']
    if engineName in csoundengine.activeEngines():
        return csoundengine.getEngine(engineName)
    numChannels = numChannels or config['play.numChannels']
    if backend == "?":
        backends = [b.name for b in csoundengine.csoundlib.audioBackends(available=True)]
        backend = tools.selectFromList(backends, title="Select Backend")
    backend = backend or config['play.backend']
    logger.debug(f"Starting engine {engineName} (nchnls={numChannels})")
    return csoundengine.Engine(name=engineName,
                               nchnls=numChannels,
                               backend=backend,
                               globalcode=_prelude,
                               quiet=not config['play.verbose'],
                               latency=config['play.schedLatency'])


def stopSynths(stopengine=False, cancelfuture=True):
    """
    Stops all synths (notes, chords, etc) being played

    If stopengine is True, the play engine itself is stopped
    """
    session = getPlaySession()
    session.unschedAll(future=cancelfuture)
    if stopengine:
        getPlayEngine().stop()


def getPlaySession() -> csoundengine.Session:
    config = activeConfig()
    group = config['play.engineName']
    if not isEngineActive():
        if config['play.autostartEngine']:
            startPlayEngine()
        else:
            raise PlayEngineNotStarted("Engine is not running. Call startPlayEngine")
    return csoundengine.getSession(group)


def isEngineActive() -> bool:
    """
    Returns True if the sound engine is active
    """
    name = activeConfig()['play.engineName']
    return csoundengine.getEngine(name) is not None


def getPlayEngine(start=None) -> Optional[csoundengine.Engine]:
    """
    Return the sound engine, or None if it has not been started
    """
    cfg = activeConfig()
    engine = csoundengine.getEngine(name=cfg['play.engineName'])
    if not engine:
        logger.debug("engine not started")
        start = start if start is not None else cfg['play.autostartEngine']
        if start:
            engine = startPlayEngine()
            return engine
        return None
    return engine


class rendering:
    def __init__(self, outfile:str=None, wait=True, quiet=None,
                 sr:int=None, nchnls:int=None):
        """
        Context manager to transform all calls to .play to be renderer offline

        Args:
            outfile: events played within this context will be rendered
                to this file. If set to None, rendering is performed to an auto-generated
                file in the recordings folder
            wait: if True, wait until rendering is done
            quiet: if True, supress any output from the csound
                subprocess (config 'rec.quiet')

        Example::

            # this will generate a file foo.wav after leaving the `with` block
            with rendering("foo.wav"):
                chord.play(dur=2)
                note.play(dur=1, fade=0.1, delay=1)

            # You can render manually, if needed
            with rendering() as r:
                chord.play(dur=2)
                ...
                print(r.getCsd())
                r.render("outfile.wav")

        """
        self.sr = sr
        self.nchnls = nchnls
        self.outfile = outfile
        self._oldRenderer: Optional[OfflineRenderer] = None
        self.renderer: Optional[OfflineRenderer] = None
        self.quiet = quiet or activeConfig()['rec.quiet']
        self.wait = wait

    def __enter__(self):
        workspace = activeWorkspace()
        self._oldRenderer = workspace.renderer
        self.renderer = OfflineRenderer(sr=self.sr, outfile=self.outfile)
        workspace.renderer = self.renderer
        return self.renderer

    def __exit__(self, exc_type, exc_value, traceback):
        w = activeWorkspace()
        w.renderer = self._oldRenderer
        if self.outfile is None:
            self.outfile = _makeRecordingFilename()
            logger.info(f"Rendering to {self.outfile}")
        self.renderer.render(outfile=self.outfile, wait=self.wait,
                             quiet=self.quiet)


def _schedOffline(renderer: csoundengine.Renderer,
                  events: List[CsoundEvent],
                  _checkNchnls=True
                  ) -> None:
    """
    Schedule the given events for offline rendering.

    You need to call renderer.render(...) to actually render/play the
    scheduled events

    Args:
        renderer: a Renderer as returned by makeRenderer
        events: events as returned by, for example, chord.events(**kws)
        _checkNchnls: (internal parameter)
            if True, will check (and adjust) nchnls in
            the renderer so that it is high enough for all
            events to render properly
    """
    if _checkNchnls:
        maxchan = max(presetManager.eventMaxNumChannels(event)
                      for event in events)
        if renderer.nchnls < maxchan:
            logger.info(f"_schedOffline: the renderer was defined with "
                        f"nchnls={renderer.csd.nchnls}, but {maxchan} "
                        f"are needed to render the given events. "
                        f"Setting nchnls to {maxchan}")
            renderer.csd.nchnls = maxchan
    for event in events:
        pargs = event.getPfields()
        if pargs[2] != 0:
            logger.warn(f"got an event with a tabnum already set...: {pargs}")
            logger.warn(f"event: {event}")
        instrName = event.instr
        assert instrName is not None
        presetdef = presetManager.getPreset(instrName)
        instr = presetdef.makeInstr()
        if not renderer.isInstrDefined(instr.name):
            renderer.registerInstr(presetdef.makeInstr())
        # renderer.defInstr(instrName, body=presetdef.body, tabledef=presetdef.params)
        renderer.sched(instrName, delay=pargs[0], dur=pargs[1],
                       pargs=pargs[3:],
                       tabargs=event.namedArgs,
                       priority=event.priority)


def playEvents(events: List[CsoundEvent],
               ) -> csoundengine.synth.SynthGroup:
    """
    Play a list of events

    Args:
        events: a list of CsoundEvents

    Returns:
        A SynthGroup

    Example::

        from maelzel.core import *
        group = Group([
            Note("4G", dur=8),
            Chord("4C 4E", dur=7, start=1)
            Note("4C#", start=1.5, dur=6)])
        play.playEvents(group.events(instr='.piano')
    """
    synths = []
    session = getPlaySession()
    presetNames = {ev.instr for ev in events}
    presetDefs = [presetManager.getPreset(name) for name in presetNames]
    presetToInstr: Dict[str, csoundengine.Instr] = {preset.name:_registerPresetInSession(preset, session)
                                                    for preset in presetDefs}
    # We take a reference time before starting scheduling,
    # so we can guarantee that events which are supposed to be
    # in sync, are in fact in sync. We could use Engine.lockReferenceTime
    # but we might interfere with another called doing the same.
    elapsed = session.engine.elapsedTime() + session.engine.extraLatency
    for ev in events:
        instr = presetToInstr[ev.instr]
        args = ev.getPfields(numchans=instr.numchans)
        synth = session.sched(instr.name,
                              delay=args[0]+elapsed,
                              dur=args[1],
                              pargs=args[3:],
                              tabargs=ev.namedArgs,
                              priority=ev.priority,
                              relative=False)
        synths.append(synth)
    return csoundengine.synth.SynthGroup(synths)
