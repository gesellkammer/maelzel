"""
This module handles playing of events

"""
from __future__ import annotations
import os

from datetime import datetime

import emlib.misc

from ._common import logger
from . import _util
from . import _dialogs
from .presetbase import *
from .presetman import presetManager, _csoundPrelude as _prelude
from .errors import *
from .workspace import getConfig, getWorkspace
import csoundengine
from .csoundevent import CsoundEvent

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union, Optional
    from .musicobjbase import MusicObj
    from maelzel.snd import audiosample
    import subprocess


__all__ = (
    'OfflineRenderer',
    'rendering',
    'lockedClock',
    'testAudio'
)


_invalidVariables = {"kfreq", "kamp", "kpitch"}


class OfflineRenderer:
    """
    An OfflineRenderer is created to render musical objects to a soundfile

    .. admonition:: OfflineRenderer as context manager

        The simplest way to render offline is to use an OfflineRenderer
        as a context manager (see also :func:`rendering`). Within this
        context any .play call will be collected and everything will be
        rendered when exiting the context
        (:ref:`see example below <offlineRendererExample>`)

    Args:
        outfile: the path to the rendered soundfile. If not given, a path
            within the record path [1]_ is returned
        sr: the samplerate of the render (:ref:`config key: 'rec.sr' <config_rec_sr>`)
        ksmps: the ksmps used for the recording
        quiet: if True, debugging output is minimized. If None, defaults to
            config (:ref:`key: 'rec.quiet' <config_rec_quiet>`)

    If rendering offline in tandem with audio samples and other csoundengine's
    functionality, it is possible to access the underlying csoundengine's Renderer
    via the ``.renderer`` attribute

    .. [1] To get the current *record path*: ``getWorkspace().recordPath()``

    (see :meth:`~maelzel.core.workspace.Workspace.recordPath`)

    .. _offlineRendererExample:

    Example
    ~~~~~~~

    Render a chromatic scale in sync with a soundfile

        >>> from maelzel.core import *
        >>> import sndfileio
        >>> notes = [Note(n, dur=0.5) for n in range(48, 72)]
        >>> chain = Chain(notes)
        >>> defPresetSoundfont('piano', sf2path='/path/to/piano.sf2')
        >>> samples, sr = sndfileio.sndread('/path/to/soundfile')
        >>> with rendering('scale.wav') as r:
        ...     chain.play(instr='piano')
        ...     r.renderer.playSample((samples, sr))

    When exiting the context manager the file 'scale.wav' is rendered. During
    the context manager, all calls to .play are intersected and scheduled
    via the OfflineRenderer
    """
    def __init__(self, outfile: str = None, sr=None, ksmps=64, nchnls=2, quiet:bool=None):
        w = getWorkspace()
        cfg = getConfig()

        self._outfile = outfile
        """Outfile given for rendering"""

        self.a4 = w.a4
        """A value for the reference frequency"""

        self.sr = sr or cfg['rec.sr']
        """The samplerate. If not given, ['rec.sr'] is used """

        self.ksmps = ksmps
        """ksmps value (samples per block)"""

        self.renderer: csoundengine.Renderer = presetManager.makeRenderer(sr, ksmps=ksmps, nchnls=nchnls)
        """The actual csoundengine.Renderer"""

        self.events: List[CsoundEvent] = []
        """A list of all events rendered"""

        self.instrDefs: Dict[str, csoundengine.Instr] = {}
        """An index of registered Instrs"""

        self.renderedSoundfiles: List[str] = []
        """A list of soundfiles rendered with this renderer"""

        self._quiet = quiet

        self._renderProc: Optional[subprocess.Popen] = None

    def __repr__(self):
        return f"OfflineRenderer(sr={self.sr})"

    def _repr_html_(self) -> str:
        sndfile = self.lastOutfile()
        if not sndfile:
            return f'<strong>OfflineRenderer</strong>(sr={self.sr})'
        from maelzel import colortheory
        blue = colortheory.safeColors['blue1']
        if not os.path.exists(sndfile):
            info = f'lastOutfile=<code style="color:{blue}">"{sndfile}"</code>'
            return f'<strong>OfflineRenderer</strong>({info})'
        from maelzel.snd import audiosample
        sample = audiosample.Sample(sndfile)
        samplehtml = sample.reprHtml(withHeader=False, withAudiotag=True)
        header = f'<strong>OfflineRenderer</strong>'

        def _(s):
            return f'<code style="color:{blue}">{s}</code>'

        sndfilestr = f'"{sndfile}"'
        info = f'outfile={_(sndfilestr)}, {_(sample.numchannels)} channels, ' \
               f'{_(format(sample.duration, ".2f"))} secs, {_(sample.sr)} Hz'
        header = f'{header}({info})'
        return '<br>'.join([header, samplehtml])

    def registerInstr(self, instrname: str, instrdef: csoundengine.Instr) -> None:
        """
        Register a csoundengine.Instr to be used with this OfflineRenderer

        .. note::

            All :class:`csoundengine.Instr` defined in the play Session are
            available to be rendered offline without the need to be registered

        Args:
            instrname: the name of this preset/instrument
            instrdef: the csoundengine.Instr instance

        """
        self.instrDefs[instrname] = instrdef
        self.renderer.registerInstr(instrdef)

    def play(self, obj: MusicObj, **kws) -> List[csoundengine.offline.ScoreEvent]:
        """
        Schedule the events generated by this obj to be renderer offline

        Args:
            obj: the object to be played offline
            kws: any keyword passed to the .events method of the obj

        Returns:
            the offline score events
        """
        events = obj.events(**kws)
        scoreEvents = [self.schedEvent(ev) for ev in events]
        return scoreEvents

    def schedEvent(self, event: CsoundEvent) -> csoundengine.offline.ScoreEvent:
        """
        Schedule a CsoundEvent as returned by :meth:`MusicObj.events() <maelzel.core.musicobj.MusicObj.events>`

        Args:
            event: a :class:`~maelzel.core.csoundevent.CsoundEvent`

        Returns:
            a ScoreEvent

        See Also:
            sched
        """
        # NB: the instrname is actually the preset name.
        instrname = event.instr
        instrdef = self.instrDefs.get(instrname)
        if instrdef is None:
            preset = presetManager.getPreset(instrname)
            if not preset:
                raise ValueError(f"Unknown instr: {instrname}")
            instrdef = preset.getInstr()
            self.registerInstr(instrname, instrdef)
        args = event.resolvePfields(instrdef)
        return self.renderer.sched(instrdef.name, delay=event.delay, dur=event.dur,
                                   pargs=args[3:], priority=event.priority,
                                   tabargs=event.namedArgs)

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

        Schedule a reverb at a higher priority to affect all notes played. Notice
        that the reverb instrument is declared at the play Session (see
        :func:`play.getPlaySession() <maelzel.core.play.getPlaySession>`). All instruments
        registered at this Session are immediately available for offline rendering.

            >>> from maelzel.core import *
            >>> scale = Chain([Note(n) for n in "4C 4D 4E 4F 4G".split()])
            >>> play.getPlaySession().defInstr('reverb', r'''
            ... |kfeedback=0.6|
            ... amon1, amon2 monitor
            ... a1, a2 reverbsc amon1, amon2, kfeedback, 12000, sr, 0.6
            ... outch 1, a1-amon1, 2, a2-amon2
            ... ''')
            >>> presetManager.defPresetSoundfont('piano', '/path/to/piano.sf2')
            >>> with play.OfflineRenderer() as r:
            ...     r.sched('reverb', priority=2)
            ...     scale.play('piano')

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

    def render(self, outfile:str=None, wait=None, quiet=None, openWhenDone=False,
               compressionBitrate: int = None
               ) -> str:
        """
        Render the events scheduled until now.

        You can access the rendering subprocess (a :class:`subprocess.Popen` object)
        via :meth:`~OfflineRenderer.lastRenderProc`

        Args:
            outfile: the soundfile to generate. Use "?" to save via a GUI dialog,
                None will render to a temporary file
            wait: if True, wait until rendering is done
            quiet: if True, supress all output generated by csound itself
                (print statements and similar opcodes still produce output)
            openWhenDone: if True, open the rendered soundfile in the default
                application

        Returns:
            the path of the rendered file

        Example
        ~~~~~~~

        TODO
        """
        self._renderProc = None
        cfg = getConfig()
        if outfile is None:
            outfile = self._outfile
        if outfile == '?':
            outfile = _dialogs.saveRecordingDialog()
            if not outfile:
                raise CancelledError("Render operation was cancelled")
        elif not outfile:
            outfile = _makeRecordingFilename(ext=".wav")
        outfile = _util.normalizeFilename(outfile)
        self.renderedSoundfiles.append(outfile)
        if quiet is None:
            quiet = self._quiet if self._quiet is not None else cfg['rec.quiet']
        if wait is None:
            wait = cfg['rec.block']
        if compressionBitrate is None:
            compressionBitrate = cfg['rec.compressionBitrate']
        outfile, proc = self.renderer.render(outfile=outfile, wait=wait, quiet=quiet,
                                             openWhenDone=openWhenDone, compressionBitrate=compressionBitrate)
        self._renderProc = proc
        return outfile

    def openLastOutfile(self) -> None:
        """
        Open last rendered outfile in an external app

        Will do nothing if there is no outfile. If the render is in progress
        this operation will block.
        """
        lastoutfile = self.lastOutfile()
        if not lastoutfile:
            return
        if not os.path.exists(lastoutfile):
            lastproc = self.lastRenderProc()
            if not lastproc:
                return
            if lastproc.poll() is None:
                lastproc.wait()
        assert os.path.exists(lastoutfile)
        emlib.misc.open_with_app(lastoutfile)

    def lastOutfile(self) -> Optional[str]:
        """
        Last rendered outfile, None if no soundfiles were rendered

        Example
        ~~~~~~~

            >>> r = OfflineRenderer(...)
            >>> r.sched(...)
            >>> r.render(wait=True)
            >>> r.lastOutfile()
            '~/.local/share/maelzel/recordings/tmpsasjdas.wav'
        """
        return self.renderedSoundfiles[-1] if self.renderedSoundfiles else None

    def lastRenderProc(self) -> Optional[subprocess.Popen]:
        """
        Last process (subprocess.Popen) used for rendering

        Example
        ~~~~~~~

            >>> r = OfflineRenderer(...)
            >>> r.sched(...)
            >>> r.render("outfile.wav", wait=False)
            >>> if (proc := r.lastRenderProc()) is not None:
            ...     proc.wait()
            ...     print(proc.stdout.read())

        """
        return self._renderProc

    def getCsd(self) -> str:
        """
        Return the .csd as string
        """
        return self.renderer.generateCsdString()

    def writeCsd(self, outfile: str = '?') -> str:
        """
        Write the .csd which would render all events scheduled until now

        Args:
            outfile: the path of the saved .csd

        Returns:
            the outfile
        """
        if outfile == "?":
            outfile = _dialogs.selectFileForSave("saveCsdLastDir", filter="Csd (*.csd)")
            if not outfile:
                raise CancelledError("Save operation cancelled")
        self.renderer.writeCsd(outfile)
        return outfile

    def __enter__(self):
        workspace = getWorkspace()
        self._oldRenderer = workspace.renderer
        workspace.renderer = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            # There was an exception since entering
            logger.warning("Offline rendering aborted")
            return
        w = getWorkspace()
        w.renderer = self._oldRenderer
        outfile = self._outfile or _makeRecordingFilename()
        logger.info(f"Rendering to {outfile}")
        self.render(outfile=outfile, wait=True)

    def renderedSample(self) -> audiosample.Sample:
        """
        Returns the last rendered soundfile as a :class:`maelzel.snd.audiosample.Sample`
        """
        assert self.renderedSoundfiles
        lastsnd = self.renderedSoundfiles[-1]
        assert os.path.exists(lastsnd)
        from maelzel.snd import audiosample
        return audiosample.Sample(lastsnd)

    def openRenderedSoundfile(self) -> None:
        """
        Opens the rendered soundfile in the default external app
        """
        if not self.renderedSoundfiles:
            raise RuntimeError("No soundfile rendered yet")
        sndfile = self.renderedSoundfiles[-1]
        if not os.path.exists(sndfile):
            raise RuntimeError(f"Did not find rendered file {sndfile}")
        import emlib.misc
        emlib.misc.open_with_app(sndfile)

    def isRendering(self) -> bool:
        """
        True if still rendering

        Returns:
            True if rendering is still in course
        """
        proc = self.lastRenderProc()
        return proc.poll() is None

    def wait(self, timeout=0) -> None:
        """
        Wait until finished rendering

        Args:
            timeout: a timeout (0 to wait indefinitely)

        """
        proc = self.lastRenderProc()
        if proc.poll() is None:
            proc.wait(timeout=timeout)


def rendering(outfile: str = None, sr=None, nchnls=2,
              fmt: str = None, **kws
              ) -> OfflineRenderer:
    """
    Creates a **context manager** to render any .play call offline

    Args:
        outfile: the soundfile to render. None to render to a soundfile in the recordings path,
            '?' to open a "save to" dialog.
        sr: the sample rate
        nchnls: the number of channels to render to
        fmt: file format to render to. If outfile is given, the file format given by the
            extension is used. If None,
        **kws: any keywords are passed directly to :class:`OfflineRenderer`

    Returns:
        an OfflineRenderer

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
        >>> presetManager.defPresetSoundfont('piano', '/path/to/piano.sf2')
        >>> with rendering('out.wav') as r:
        ...     scale.play('piano')
        ...     r.sched('reverb', priority=2)
    """
    return OfflineRenderer(outfile=outfile, sr=sr, nchnls=nchnls, **kws)


def recEvents(events: List[CsoundEvent], outfile:str=None,
              sr:int=None, wait:bool=None, ksmps: int = None,
              quiet: bool = None, nchnls: int = None
              ) -> OfflineRenderer:
    """
    Record the events to a soundfile

    Args:
        events: a list of events as returned by .events(...)
        outfile: the generated file. If left unset, a file inside the recording
            path is created (see `recordPath`). Use "?" to save via a GUI dialog
        sr: sample rate of the soundfile
        ksmps: number of samples per cycle (:ref:`config 'rec.ksmps' <config_rec_ksmps>`)
        wait: if True, wait until recording is finished. If None,
            use the :ref:`config 'rec.block' <config_rec_block>`
        quiet: if True, supress debug information when calling
            the csound subprocess

    Returns:
        the :class:`OfflineRenderer` used to render the events. If the outfile
        was not given, the path of the recording can be retrieved from
        ``renderer.outfile``

    Example
    ~~~~~~~

    .. code-block:: python

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
    if nchnls is None:
        nchnls = max(round(ev.position + ev.chan) for ev in events)
    offlineRenderer = OfflineRenderer(sr=sr, ksmps=ksmps, nchnls=nchnls)
    for ev in events:
        offlineRenderer.schedEvent(ev)
    offlineRenderer.render(outfile=outfile, wait=wait, quiet=quiet)
    return offlineRenderer


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
        The file will be created inside the recording path
        (see :meth:`Workspace.recordPath() <maelzel.core.workspace.Workspace.recordPath>`)
    """
    path = getWorkspace().recordPath()
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
    instr = preset.getInstr()
    # registerInstr checks itself if the instr is already defined
    session.registerInstr(instr)
    return instr


def _soundfontToTabname(sfpath: str) -> str:
    path = os.path.abspath(sfpath)
    return f"gi_sf2func_{hash(path)%100000}"


def _soundfontToChannel(sfpath:str) -> str:
    basename = os.path.split(sfpath)[1]
    return f"_sf:{basename}"


def testAudio(duration=4, period=0.5, numChannels: int = None, delay=0.5,
              backend: str = None
              ) -> None:
    """
    Test the audio engine by sending pink to each channel

    Args:
        duration: the duration of the test
        period: how long to test each channel. Channels are cycled
        numChannels: the number of channels to use if starting the engine
        delay: how long to wait before starting the test.

    """
    engine = getPlayEngine(start=False)
    if not engine:
        logger.info("Starting engine...")
        engine = startPlayEngine(numChannels=numChannels, backend=backend)
    engine.testAudio(dur=duration, period=period, delay=delay)


def startPlayEngine(numChannels: int = None,
                    backend: str = None,
                    verbose: bool = None,
                    buffersize: int = None,
                    waitAfterStart=0.5) -> csoundengine.Engine:
    """
    Start the play engine

    If an engine is already active, nothing happens, even if the
    configuration is different. To start the play engine with a different
    configuration, stop the engine first.

    Args:
        numChannels: the number of output channels, overrides config 'play.numChannels'
        backend: the audio backend used, overrides config 'play.backend'
        verbose: if True, output debugging information
        buffersize: if given, use this as the buffer size. None to use a sensible
            default for the backend
    """
    config = getConfig()
    engineName = config['play.engineName']
    if engineName in csoundengine.activeEngines():
        return csoundengine.getEngine(engineName)
    numChannels = numChannels or config['play.numChannels']
    if backend == "?":
        backends = [b.name for b in csoundengine.csoundlib.audioBackends(available=True)]
        backend = _dialogs.selectFromList(backends, title="Select Backend")
    backend = backend or config['play.backend']
    verbose = verbose if verbose is not None else config['play.verbose']
    logger.debug(f"Starting engine {engineName} (nchnls={numChannels})")
    engine = csoundengine.Engine(name=engineName,
                                 nchnls=numChannels,
                                 backend=backend,
                                 globalcode=_prelude,
                                 quiet=not verbose,
                                 latency=config['play.schedLatency'],
                                 buffersize=buffersize)
    if waitAfterStart > 0:
        import time
        time.sleep(waitAfterStart)
    return engine


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
    config = getConfig()
    group = config['play.engineName']
    if not isEngineActive():
        if config['play.autostartEngine']:
            startPlayEngine()
        else:
            raise RuntimeError("Engine is not running and config['play.autostartEngine'] "
                               "is False. Call startPlayEngine")
    return csoundengine.getSession(group)


def isEngineActive() -> bool:
    """
    Returns True if the sound engine is active
    """
    name = getConfig()['play.engineName']
    return csoundengine.getEngine(name) is not None


def getPlayEngine(start=None, numChannels: int = None, **kws) -> Optional[csoundengine.Engine]:
    """
    Return the sound engine, or None if it has not been started

    Args:
        start: if True, the play Engine will be started if not started already. If
            None, the value in the config 'play.autostartEngine` (`config_play_autostartengine`)
            is used
        numChannels: only valid if the engine has  not been started and is set to be
            started (start is True or autostart is set in the config)
        kws: any keyword given will be passed to :func:`startPlayEngine` if the engine needs
            to be started

    Returns:
        the play Engine or None if the Engine has not been started

    Common keywords:

    keyword         description

    numChannels     The number of output channels of the Engine

    """
    cfg = getConfig()
    engine = csoundengine.getEngine(name=cfg['play.engineName'])
    if not engine:
        logger.debug("engine not started")
        start = start if start is not None else cfg['play.autostartEngine']
        if start:
            engine = startPlayEngine(numChannels=numChannels, **kws)
            return engine
        return None
    return engine


def lockedClock():
    """
    Context manager to schedule play events in sync.

    This is a shortcut to ``play.getPlayEngine().lockedClock()``

    Example
    -------

        >>> from maelzel.core import *
        >>> notes = [Note(m, dur=1) for m in range(60, 72)]
        >>> with play.lockedClock():
        ...     for i, n in enumerate(notes):
        ...         n.play(delay=i*0.25, instr='.piano')

    """
    return getPlayEngine().lockedClock()


def _dummySynth(dur=0.001) -> csoundengine.synth.Synth:
    engine = getPlayEngine()
    session = engine.session()
    return session.sched('.dummy', 0, dur)


def playEvents(events: List[CsoundEvent],
               whenfinished: Callable = None
               ) -> csoundengine.synth.SynthGroup:
    """
    Play a list of events

    Args:
        events: a list of CsoundEvents
        whenfinished: call this function when the last event is finished. A function taking
            no arguments and returning None

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
    presetToInstr: Dict[str, csoundengine.Instr] = {}
    sync = False
    for preset in presetDefs:
        instrdef = preset.getInstr()
        presetToInstr[preset.name] = instrdef
        if not session.isInstrRegistered(instrdef):
            sync = True
            session.registerInstr(instrdef)

    if sync:
        session.engine.sync()

    # We take a reference time before starting scheduling,
    # so we can guarantee that events which are supposed to be
    # in sync, are in fact in sync. We could use Engine.lockReferenceTime
    # but we might interfere with another called doing the same.
    elapsed = session.engine.elapsedTime() + session.engine.extraLatency
    if whenfinished:
        ev = max(events, key=lambda ev: ev.end if ev.end > 0 else float('inf'))
        ev.whenfinished = lambda id: whenfinished() if not ev.whenfinished else lambda id, ev=ev: ev.whenfinished(id) or whenfinished()
    for ev in events:
        instr = presetToInstr[ev.instr]
        args = ev.resolvePfields(instr)
        synth = session.sched(instr.name,
                              delay=args[0]+elapsed,
                              dur=args[1],
                              pargs=args[3:],
                              tabargs=ev.namedArgs,
                              priority=ev.priority,
                              relative=False,
                              whenfinished=ev.whenfinished)
        synths.append(synth)
    group = csoundengine.synth.SynthGroup(synths)
    return group


def _resolvePfields(event: CsoundEvent, instr: csoundengine.Instr
                    ) -> List[float]:
    """
    returns pfields, **beginning with p2**.

    ==== =====  ======
    idx  parg    desc
    ==== =====  ======
    0    2       delay
    1    3       duration
    2    4       tabnum
    3    5       bpsoffset (pfield index, starting with 1)
    4    6       bpsrows
    5    7       bpscols
    6    8       gain
    7    9       chan
    8    0       position
    9    1       fade0
    0    2       fade1
    1    3       pitchinterpol
    2    4       fadeshape
    .
    . reserved space for user pargs
    .
    ==== =====  ======

    breakpoint data

    tabnum: if 0 it is discarded and filled with a valid number later
    """
    pitchInterpolMethod = CsoundEvent.pitchinterpolToInt[event.pitchInterpolMethod]
    fadeshape = CsoundEvent.fadeshapeToInt[event.fadeShape]
    # if no userpargs, bpsoffset is 15
    numPargs5 = len(instr.pargsIndexToName)
    numBuiltinPargs = 10
    numUserArgs = numPargs5 - numBuiltinPargs
    bpsoffset = 15 + numUserArgs
    bpsrows = len(event.bps)
    bpscols = event.breakpointSize()
    pfields = [
        float(event.delay),
        event.dur,
        0,  # table index, to be filled later
    ]
    pfields5 = [
        bpsoffset, # p5, idx: 4
        bpsrows,
        bpscols,
        event.gain,
        event.chan,
        event.position,
        event.fadein,
        event.fadeout,
        pitchInterpolMethod,
        fadeshape
    ]
    print("# user args: ", numUserArgs, "len pfields5", len(pfields5))

    if event._namedArgsMethod == 'pargs' and numUserArgs > 0:
        pfields5 = instr.pargsTranslate(args=pfields5, kws=event.namedArgs)
    pfields.extend(pfields5)
    for bp in event.bps:
        pfields.extend(bp)

    assert all(isinstance(p, (int, float)) for p in pfields), [(p, type(p)) for p in pfields if not isinstance(p, (int, float))]
    return pfields


