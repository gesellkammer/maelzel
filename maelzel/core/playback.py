"""
This module handles playing of events

"""
from __future__ import annotations
import os

from datetime import datetime

import emlib.misc
import emlib.iterlib

from math import ceil

from ._common import logger, prettylog
from . import _util
from . import _dialogs
from .presetdef import *
from .presetmanager import presetManager
from .errors import *
from .workspace import getConfig, getWorkspace, Workspace
import csoundengine
from .synthevent import SynthEvent

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
    from .mobj import MObj
    from maelzel.snd import audiosample
    import subprocess


__all__ = (
    'render',
    'synchedplay',
    'play',
    'testAudio',
    'playEngine',
    'playSession'
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
        sr: the sr of the render (:ref:`config key: 'rec.sr' <config_rec_sr>`)
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
        >>> with render('scale.wav') as r:
        ...     chain.play(instr='piano')
        ...     r.renderer.playSample((samples, sr))

    When exiting the context manager the file 'scale.wav' is rendered. During
    the context manager, all calls to .play are intersected and scheduled
    via the OfflineRenderer
    """
    def __init__(self,
                 outfile: str = None,
                 sr=None,
                 ksmps=64,
                 numchannels=2,
                 quiet: bool = None):
        w = getWorkspace()
        cfg = w.config

        self._outfile = outfile
        """Outfile given for rendering"""

        self.a4 = w.a4
        """A value for the reference frequency"""

        self.sr = sr or cfg['rec.sr']
        """The sr. If not given, ['rec.sr'] is used """

        self.ksmps = ksmps
        """ksmps value (samples per block)"""

        self.renderer: csoundengine.Renderer = presetManager.makeRenderer(sr, ksmps=ksmps, numChannels=numchannels)
        """The actual csoundengine.Renderer"""

        #self.events: list[SynthEvent] = []
        #"""A list of all events rendered"""

        self.instrDefs: dict[str, csoundengine.Instr] = {}
        """An index of registered Instrs"""

        self.renderedSoundfiles: list[str] = []
        """A list of soundfiles rendered with this renderer"""

        self._quiet = quiet

        self._renderProc: subprocess.Popen|None = None

    @property
    def scheduledEvents(self) -> dict[int, csoundengine.offline.ScoreEvent]:
        """The scheduled events"""
        return self.renderer.scheduledEvents

    def timeRange(self) -> tuple[float, float]:
        """
        The time range of the scheduled events

        Returns:
            a tuple (start, end)
        """
        events = self.scheduledEvents.values()
        start = min(ev.start for ev in events)
        end = max(ev.end for ev in events)
        return start, end

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

    def play(self, obj: MObj, **kws) -> list[csoundengine.offline.ScoreEvent]:
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

    def schedEvent(self, event: SynthEvent) -> csoundengine.offline.ScoreEvent:
        """
        Schedule a SynthEvent as returned by
        :meth:`MObj.events() <maelzel.core.MObj.events>`

        Args:
            event: a :class:`~maelzel.core.synthevent.SynthEvent`

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
                raise ValueError(f"Unknown preset instr: {instrname}")
            instrdef = preset.getInstr()
            self.registerInstr(instrname, instrdef)
        args = event.resolvePfields(instrdef)
        return self.renderer.sched(instrdef.name, delay=event.delay, dur=event.dur,
                                   args=args[3:], priority=event.priority,
                                   tabargs=event.args)

    def schedEvents(self, events: list[SynthEvent]) -> list[csoundengine.offline.ScoreEvent]:
        """
        Schedule multiple events as returned by :meth:`MObj.events() <maelzel.core.MObj.events>`

        Args:
            events: the events to schedule

        Returns:
            a list of :class:`ScoreEvent`

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> scale = Chain([Note(m, 0.5) for m in range(60, 72)])
            >>> renderer = OfflineRenderer()
            >>> renderer.schedEvents(scale.events(instr='piano'))
            >>> renderer.render('outfile.wav')
        """
        out = [self.schedEvent(ev)
               for ev in events]
        return out

    def definedInstrs(self) -> dict[str, csoundengine.Instr]:
        """
        Get all instruments available within this OfflineRenderer

        All presets and all extra intruments registered at the active
        Session (as returned via getPlaySession) are available

        Returns:
            dict {instrname: csoundengine.Instr} with all instruments available

        """
        instrs = {}
        instrs.update(self.renderer.registeredInstrs())
        instrs.update(playSession().registeredInstrs())
        return instrs

    def sched(self,
              instrname: str,
              delay=0.,
              dur=-1.,
              priority=1,
              args: list[float] | dict[str, float] = None,
              tabargs: dict[str, float] = None,
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
            args: any pfields passed to the instr., starting at p5
            tabargs: table args accepted by the instr.
            **kws: named pfields

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
            >>> playback.playSession().defInstr('reverb', r'''
            ... |kfeedback=0.6|
            ... amon1, amon2 monitor
            ... a1, a2 reverbsc amon1, amon2, kfeedback, 12000, sr, 0.6
            ... outch 1, a1-amon1, 2, a2-amon2
            ... ''')
            >>> presetManager.defPresetSoundfont('piano', '/path/to/piano.sf2')
            >>> with playback.OfflineRenderer() as r:
            ...     r.sched('reverb', priority=2)
            ...     scale.play('piano')

        """
        if not self.renderer.isInstrDefined(instrname):
            session = playSession()
            instr = session.getInstr(instrname)
            if not instr:
                logger.error(f"Unknown instrument {instrname}. "
                             f"Defined instruments: {self.renderer.registeredInstrs().keys()}")
                raise ValueError(f"Instrument {instrname} unknown")
            self.renderer.registerInstr(instr)
        return self.renderer.sched(instrname=instrname, delay=delay, dur=dur,
                                   priority=priority, args=args,
                                   tabargs=tabargs,
                                   **kws)

    def render(self,
               outfile='',
               wait: bool | None = None,
               quiet: bool | None = None,
               openWhenDone=False,
               compressionBitrate: int | None = None,
               endtime=0.
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

            >>> from maelzel.core import *
            >>> scale = Chain([Note(n) for n in "4C 4D 4E 4F 4G".split()])
            >>> playback.playSession().defInstr('reverb', r'''
            ... |kfeedback=0.6|
            ... amon1, amon2 monitor
            ... a1, a2 reverbsc amon1, amon2, kfeedback, 12000, sr, 0.6
            ... outch 1, a1-amon1, 2, a2-amon2
            ... ''')
            >>> presetManager.defPresetSoundfont('piano', '/path/to/piano.sf2')
            >>> renderer = playback.OfflineRenderer()
            >>> renderer.schedEvents(scale.events(instr='piano'))
            >>> renderer.sched('reverb', priority=2)
            >>> renderer.render('outfile.wav')

        """
        self._renderProc = None
        cfg = getConfig()
        if not outfile:
            outfile = self._outfile
        if outfile == '?':
            outfile = _dialogs.saveRecordingDialog()
            if not outfile:
                raise CancelledError("Render operation was cancelled")
        elif not outfile:
            outfile = _makeRecordingFilename(ext=".wav")
        outfile = _util.normalizeFilename(outfile)
        if quiet is None:
            quiet = self._quiet if self._quiet is not None else cfg['rec.quiet']
        if wait is None:
            wait = cfg['rec.blocking']
        if compressionBitrate is None:
            compressionBitrate = cfg['rec.compressionBitrate']
        outfile, proc = self.renderer.render(outfile=outfile, wait=wait, quiet=quiet,
                                             openWhenDone=openWhenDone,
                                             compressionBitrate=compressionBitrate,
                                             endtime=endtime)
        self.renderedSoundfiles.append(outfile)
        self._renderProc = proc
        return outfile

    def openLastOutfile(self, timeout=None) -> str:
        """
        Open last rendered soundfile in an external app

        Will do nothing if there is no outfile. If the render is in progress
        this operation will block.

        Args:
            timeout: if the render is not finished this operation will block with the
                given timeout

        Returns:
            the path of the soundfile or an empty string if no soundfile was rendered
        """
        lastoutfile = self.lastOutfile()
        if not lastoutfile:
            logger.info(f"There are no rendered soundfiles in this {type(self).__name__}")
            return ''
        if not os.path.exists(lastoutfile):
            lastproc = self.lastRenderProc()
            if not lastproc:
                raise RuntimeError(f"The soundfile {lastoutfile} was not found but there is"
                                   f"no rendering process...")
            if lastproc.poll() is None:
                logger.debug(f"Still rendering {lastoutfile}, waiting...")
                lastproc.wait(timeout=timeout)
            assert os.path.exists(lastoutfile), "The process has finished but the soundfile " \
                                                "was not found."
        emlib.misc.open_with_app(lastoutfile)

    def lastOutfile(self) -> str|None:
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

    def lastRenderProc(self) -> subprocess.Popen|None:
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


def render(outfile: str = None,
           events: list[SynthEvent | MObj | list[MObj | SynthEvent]] = None,
           sr: int = None,
           wait: bool = None,
           ksmps: int = None,
           quiet: bool = None,
           nchnls: int = None,
           workspace: Workspace = None,
           extratime=0.,
           **kws
           ) -> OfflineRenderer:
    """
    Render to a soundfile / creates a **context manager** to render offline

    When not used as a context manager the events / objects must be given. The soundfile
    will be generated immediately.

    When used as a context manager events should be left unset. Within this context any
    call to :meth:`MObj.play` will be redirected to the offline renderer and at
    the exit of the context all events will be rendered to a soundfile.

    Args:
        outfile: the generated file. If None, a file inside the recording
            path is created (see `recordPath`). Use "?" to save via a GUI dialog or
        events: the events/objects to play. This can only be left unset if using ``render``
            as a context manager (see example)
        sr: sample rate of the soundfile (:ref:`config 'rec.sr' <config_rec_sr>`)
        ksmps: number of samples per cycle (:ref:`config 'rec.ksmps' <config_rec_ksmps>`)
        wait: if True, wait until recording is finished. If None,
            use the :ref:`config 'rec.blocking' <config_rec_blocking>`
        quiet: if True, supress debug information when calling
            the csound subprocess
        extratime: extra time added at the end of the render to allow

    Returns:
        the :class:`OfflineRenderer` used to render the events. If the outfile
        was not given, the path of the recording can be retrieved from
        ``renderer.outfile``

    Example
    ~~~~~~~

        >>> a = Chord("A4 C5", start=1, dur=2)
        >>> b = Note("G#4", dur=4)
        >>> events = sum([
        ...     a.events(chan=1),
        ...     b.events(chan=2, gain=0.2)
        ... ], [])
        >>> render("out.wav", events)

    This can be used also as a context manager (in this case events must be None):

        >>> from maelzel.core import *
        >>> scale = Chain([Note(n) for n in "4C 4D 4E 4F 4G".split()])
        >>> playback.playSession().defInstr('reverb', r'''
        ... |kfeedback=0.6|
        ... amon1, amon2 monitor
        ... a1, a2 reverbsc amon1, amon2, kfeedback, 12000, sr, 0.6
        ... outch 1, a1-amon1, 2, a2-amon2
        ... ''')
        >>> presetManager.defPresetSoundfont('piano', '/path/to/piano.sf2')
        >>> with render() as r:
        ...     scale.play('piano')
        ...     r.sched('reverb', priority=2)



    See Also
    ~~~~~~~~

    :class:`OfflineRenderer`
    """
    if events:
        events, sessionevents = _collectEvents(events, eventparams=kws, workspace=workspace)
        return _recEvents(events=events, outfile=outfile, sr=sr, wait=wait,
                          ksmps=ksmps, quiet=quiet, numchannels=nchnls,
                          extratime=extratime)
    else:
        return OfflineRenderer(outfile=outfile, sr=sr, numchannels=nchnls, quiet=quiet, ksmps=ksmps)


def _recEvents(events: list[SynthEvent],
               sessionevents: list[csoundengine.session.SessionEvent] = None,
               outfile: str = None,
               sr: int = None,
               wait: bool = None,
               ksmps: int = None,
               quiet: bool = None,
               numchannels: int = None,
               extratime: float = 0.
               ) -> OfflineRenderer:
    """
    Record the events to a soundfile

    Args:
        events: a flat list of events
        outfile: the generated file. If left unset, a file inside the recording
            path is created (see `recordPath`). Use "?" to save via a GUI dialog
        sr: sample rate of the soundfile (:ref:`config 'rec.sr' <config_rec_sr>`)
        ksmps: number of samples per cycle (:ref:`config 'rec.ksmps' <config_rec_ksmps>`)
        wait: if True, wait until recording is finished. If None,
            use the :ref:`config 'rec.blocking' <config_rec_blocking>`
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
    assert events or sessionevents, "Nothin to render"
    if not numchannels:
        numchannels = max(int(ceil(ev.position + ev.chan)) for ev in events)
        # nchnls = max(nchnls, getConfig()['rec.numChannels'])
    # endtime = max(ev.end for ev in events)
    renderer = OfflineRenderer(sr=sr, ksmps=ksmps, numchannels=numchannels)
    for ev in events:
        renderer.schedEvent(ev)
    if sessionevents:
        for ev in sessionevents:
            renderer.sched(instrname=ev.instrname,
                           delay=ev.delay,
                           dur=ev.dur,
                           priority=ev.priority,
                           args=ev.args,
                           tabargs=ev.tabargs)
    if extratime:
        starttime, endtime = renderer.timeRange()
        endtime += extratime
    else:
        endtime = 0.
    renderer.render(outfile=outfile, wait=wait, quiet=quiet, endtime=endtime)
    return renderer


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
    engine = playEngine(numchannels=numChannels, backend=backend)
    if not engine:
        logger.info("Starting engine...")
        engine = playEngine(numchannels=numChannels, backend=backend)
    engine.testAudio(dur=duration, period=period, delay=delay)


def getAudioDevices(backend: str = None
                    ) -> tuple[list[csoundengine.csoundlib.AudioDevice],
                               list[csoundengine.csoundlib.AudioDevice]]:
    """
    Returns (indevices, outdevices), where each of these lists is an AudioDevice.

    Args:
        backend: specify a backend supported by your installation of csound.
            None to use a default for you OS

    Returns:
        a tuple of (input devices, output devices)

    .. note::

        For jack an audio device is a client

    An AudioDevice is defined as::

        AudioDevice:
            id: str
            name: str
            kind: str
            index: int = -1
            numchannels: Optional[int] = 0

    Of is the ``.name`` attribute: this is the value that needs to be
    passed to :func:`playEngine` to select a specific device.

    .. seealso:: :func:`playEngine`
    """
    return csoundengine.csoundlib.getAudioDevices(backend=backend)


def playEngine(numchannels: int = None,
               backend: str = None,
               outdev: str = None,
               verbose: bool = None,
               buffersize: int = None,
               latency: float = None,
               ) -> csoundengine.Engine:
    """
    Get the play engine; start it if needed

    **maelzel** used csound for any sound related task (synthesis, sample
    playback, etc). To interact with csound it relies on the
    :class:`Engine <https://csoundengine.readthedocs.io/en/latest/api/csoundengine.engine.Engine.html>` class

    If an Engine is already active, that Engine is returned, even if the configuration
    is different. **To start the play engine with a different configuration, stop the engine first**

    Args:
        numchannels: the number of output channels, overrides config 'play.numChannels'
        backend: the audio backend used, overrides config 'play.backend'
        outdev: leave as None to use the backend's default, use '?' to select
            from a list of available devices. To list all available devices
            see :func:`getAudioDevices`
        verbose: if True, output debugging information
        buffersize: if given, use this as the buffer size. None to use a sensible
            default for the backend
        latency: an added latency

    Returns:
        the play Engine


    .. seealso:: :func:`getAudioDevices`
    """
    config = Workspace.active.config
    engineName = config['play.engineName']
    if engine := csoundengine.getEngine(engineName):
        if any(_ is not None for _ in (numchannels, backend, outdev, verbose, buffersize, latency)):
            prettylog('WARNING',
                      "\nThe sound engine has been started already. Any configuration passed "
                      f"will have no effect. To modify the configuration of the engine first "
                      f"stop the engine (`playEngine().stop()`) and call `playEngine(...)` "
                      f"with the desired configuration. "
                      f"\nCurrent sound engine: {engine}")
        return engine
    numchannels = numchannels or config['play.numChannels']
    if backend == "?":
        backends = [b.name for b in csoundengine.csoundlib.audioBackends(available=True)]
        backend = _dialogs.selectFromList(backends, title="Select Backend")
    backend = backend or config['play.backend']
    verbose = verbose if verbose is not None else config['play.verbose']
    logger.debug(f"Starting engine {engineName} (nchnls={numchannels})")
    latency = latency if latency is not None else config['play.schedLatency']
    engine = csoundengine.Engine(name=engineName,
                                 nchnls=numchannels,
                                 backend=backend,
                                 outdev=outdev,
                                 globalcode=presetManager.csoundPrelude,
                                 quiet=not verbose,
                                 latency=latency,
                                 buffersize=buffersize,
                                 a4=config['A4'])
    waitAfterStart = config['play.waitAfterStart']
    if waitAfterStart > 0:
        import time
        time.sleep(waitAfterStart)
    # We create the session as soon as possible, to configure the engine for
    # the session's reserved instrument ranges / tables
    _ = engine.session()
    return engine


def stopSynths(stopengine=False, cancelfuture=True):
    """
    Stops all synths (notes, chords, etc) being played

    If stopengine is True, the play engine itself is stopped
    """
    session = playSession()
    session.unschedAll(future=cancelfuture)
    if stopengine:
        playEngine().stop()


def playSession() -> csoundengine.Session|None:
    """Returns the csoundengine.Session managing the running Engine

    .. seealso:: :class:`csoundengine.Session <https://csoundengine.readthedocs.io/en/latest/api/csoundengine.session.Session.html>`
    """
    engine = playEngine()
    return engine.session() if engine else None


def isEngineActive() -> bool:
    """
    Returns True if the sound engine is active
    """
    name = getConfig()['play.engineName']
    return csoundengine.getEngine(name) is not None


def _dummySynth(dur=0.001) -> csoundengine.synth.Synth:
    engine = playEngine()
    session = engine.session()
    return session.sched('.dummy', 0, dur)


def _collectEvents(events: Sequence[MObj | Sequence[SynthEvent]],
                   eventparams: dict,
                   workspace: Workspace
                   ) -> tuple[list[SynthEvent], list[csoundengine.session.SessionEvent]]:
    maelzelevents = []
    sessionevents = []
    if workspace is None:
        workspace = Workspace.active
    for ev in events:
        if isinstance(ev, (list, tuple)):
            if isinstance(ev[0], SynthEvent):
                maelzelevents.extend(ev)
            else:
                evs, sessionevs = _collectEvents(ev, eventparams=eventparams, workspace=workspace)
                maelzelevents.extend(evs)
                sessionevents.extend(sessionevs)
        elif isinstance(ev, SynthEvent):
            maelzelevents.append(ev)
        elif isinstance(ev, csoundengine.session.SessionEvent):
            sessionevents.append(ev)
        else:
            maelzelevents.extend(ev.events(workspace=workspace, **eventparams))
    return maelzelevents, sessionevents


def play(*sources: MObj | Sequence[SynthEvent] | csoundengine.session.SessionEvent,
         whenfinished: Callable = None,
         workspace: Workspace = None,
         **eventparams
         ) -> csoundengine.synth.SynthGroup:
    """
    Play a sequence of objects / events

    When playing multiple objects via their respective .play method, initialization
    (loading soundfiles, soundfonts, etc) might result in events getting out of sync
    with each other.

    This function first collects all events; any initialization is done beforehand
    as to ensure that events keep in sync. After initialization all events are scheduled
    and their synths are gathered in a SynthGroup

    .. note::

        To customize playback use the ``.events`` method, which works exactly like
        ``.play`` but returns the data so that it can be played later.

    Args:
        sources: a possibly nested sequence of MObjs or events as returned from
            :meth:`MObj.events`
        whenfinished: call this function when the last event is finished. A function taking
            no arguments and returning None
        workspace: if given it will override the active workspace
        eventparams: any keyword arguments will be passed to :meth:`MObj.events` if
            events need to be generated

    Returns:
        A SynthGroup holding all scheduled synths

    Example::

        >>> from maelzel.core import *
        >>> from csoundengine.session import SessionEvent
        >>> import csoundengine as ce
        >>> session = playSession()
        >>> session.defInstr('reverb', r'''
        >>> |kfeedback=0.85|
        ... a1, a2 monitor
        ... aL, aR  reverbsc a1, a2, kfeedback, 12000, sr, 0.5, 1
        ... outch 1, aL - a1, 2, aR - a2
        ... ''')
        >>> session.defInstr('sin', r'''
        ... |imidi=60, iamp=0.01|
        ... a1 oscili iamp, mtof(imidi)
        ... a1 *= linsegr(0, 0.5, 1, 2, 0)
        ... outch 1, a1
        ... ''')
        >>> play(
        >>>     Chord("4C 4E", 7, start=1).events(position=0.5),
        >>>     Note("4C#", 6, offset=1.5),  # No customization,
        >>>     SessionEvent('reverb', dur=10, args={'kfeedback': 0.8}, priority=2),
        >>>     SessionEvent('sin', delay=0.1, dur=3, args={'imidi': 61.33, 'iamp':0.02})
        >>> )

    .. seealso:: :class:`synchedplay`
    """
    flatevents, sessionevents = _collectEvents(sources, eventparams=eventparams, workspace=workspace)
    return _playFlatEvents(flatevents, sessionevents=sessionevents, whenfinished=whenfinished)


def _prepareEvents(events: list[SynthEvent], session: csoundengine.Session, block=True
                   ) -> dict[str, csoundengine.Instr]:
    assert events
    presetNames = {ev.instr for ev in events}
    presetDefs = [presetManager.getPreset(name) for name in presetNames]
    presetToInstr: dict[str, csoundengine.Instr] = {}
    sync = False
    for preset in presetDefs:
        instrdef = preset.getInstr()
        presetToInstr[preset.name] = instrdef
        if not session.isInstrRegistered(instrdef):
            sync = True
            session.registerInstr(instrdef)
    for event in events:
        instrdef = presetToInstr[event.instr]
        session.prepareSched(instrdef.name, event.priority, block=False)
    if sync and block:
        session.engine.sync()
    return presetToInstr


def _playFlatEvents(events: list[SynthEvent],
                    sessionevents: list[csoundengine.session.SessionEvent] = None,
                    whenfinished: Callable = None,
                    presetToInstr: dict[str, csoundengine.Instr] = None
                    ) -> csoundengine.synth.SynthGroup:
    """
    Play a list of events

    Args:
        events: a list of SynthEvents
        sessionevents: if given, a list of csoundengine SessionEvents, which are events
            played directly at the csoundengine's Session level but are gathered in the
            returned synthgroup
        whenfinished: call this function when the last event is finished. A function taking
            no arguments and returning None
        presetToInstr: normally None, otherwise a dict mapping preset name to
            csoundengine's Instr as calculated within _prepareEvents. It should only be
            given if _prepareEvents was previously called for the events passed here
            and should not be called again (see synchedplay for this pattern of usage)

    Returns:
        A SynthGroup

    """
    if not isEngineActive():
        numChannels = max(int(ceil(ev.position + ev.chan)) for ev in events)
        numChannels = max(numChannels, 2)
        playEngine(numchannels=numChannels)
    session = playSession()
    if presetToInstr is None:
        presetToInstr = _prepareEvents(events, session)

    if sessionevents:
        for ev in sessionevents:
            session.prepareSched(ev.instrname, priority=ev.priority)
    synths = []

    resolvedArgs = [ev.resolvePfields(presetToInstr[ev.instr]) for ev in events]

    if whenfinished:
        ev = max(events, key=lambda ev: ev.end if ev.end > 0 else float('inf'))
        ev.whenfinished = lambda id: whenfinished() if not ev.whenfinished else lambda id, ev=ev: ev.whenfinished(id) or whenfinished()

    # We take a reference time before starting scheduling,
    # so we can guarantee that events which are supposed to be
    # in sync, are in fact in sync. We could use Engine.lockReferenceTime
    # but we might interfere with another caller doing the same.
    elapsed = session.engine.elapsedTime() + session.engine.extraLatency
    for ev, args in zip(events, resolvedArgs):
        instr = presetToInstr[ev.instr]
        synth = session.sched(instr.name,
                              delay=args[0]+elapsed,
                              dur=args[1],
                              args=args[3:],
                              tabargs=ev.args,
                              priority=ev.priority,
                              relative=False,
                              whenfinished=ev.whenfinished)
        synths.append(synth)

    if sessionevents:
        sessionsynths = [session.schedEvent(ev)
                         for ev in sessionevents]
        synths.extend(sessionsynths)
    return csoundengine.synth.SynthGroup(synths)


def _resolvePfields(event: SynthEvent, instr: csoundengine.Instr
                    ) -> list[float]:
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
    pitchInterpolMethod = SynthEvent.pitchinterpolToInt[event.pitchinterpol]
    fadeshape = SynthEvent.fadeshapeToInt[event.fadeShape]
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
    if event._namedArgsMethod == 'pargs' and numUserArgs > 0:
        pfields5 = instr.pargsTranslate(args=pfields5, kws=event.args)
    pfields.extend(pfields5)
    for bp in event.bps:
        pfields.extend(bp)

    assert all(isinstance(p, (int, float)) for p in pfields), [(p, type(p)) for p in pfields if not isinstance(p, (int, float))]
    return pfields


class synchedplay:
    """
    Context manager to group realtime events to ensure synched playback

    When playing multiple objects via their respective .play method, initialization
    (loading soundfiles, soundfonts, etc) might result in events getting out of sync
    with each other.

    Within this context all ``.play`` calls are collected and all events are
    scheduled at the end of the context. Any initialization is done beforehand
    as to ensure that events keep in sync. Pure csound events can also be
    scheduled in sync during this context, using the ``sched`` method.

    After exiting the context all scheduled synths can be
    accessed via the ``synthgroup`` attribute.

    .. note::

        Use this context manager whenever you are mixing multiple objects with
        customized play arguments, and external csoundengine instruments

    Args:
        whenfinished: call this function when the last event is finished. A function taking
            no arguments and returning None

    Example
    ~~~~~~~

        >>> from maelzel.core import *
        >>> session = playSession()  # returns a csoundengine.session.Session
        >>> session.defInstr('reverb', r'''
        ... |kfeedback=0.85|
        ... a1, a2 monitor
        ... aL, aR  reverbsc a1, a2, kfeedback, 12000, sr, 0.5, 1
        ... outch 1, aL - a1, 2, aR - a2
        ... ''')
        >>> chain = Chain([Note(m, 0.5) for m in range(60, 72)])
        >>> with synchedplay() as s:
        ...     chain.play(position=1, instr='piano')
        ...     s.sched('reverb', 0, dur=10, priority=2)
    """
    def __init__(self, whenfinished: Callable = None):
        self.lock = True
        """Should the engine's clock be locked during play?"""

        self.events: list[SynthEvent] = []
        """A list of all events rendered"""

        self.instrDefs: dict[str, csoundengine.Instr] = {}
        """An index of registered Instrs"""

        self.synthgroup: csoundengine.synth.SynthGroup | None = None
        """A SynthGroup holding all scheduled synths during the context"""

        self.engine: csoundengine.Engine
        """The play engine, can be used during the context"""

        self.session: csoundengine.Session
        """The corresponding Session, can be used to access the session during the context"""

        self._realtimeEvents: list[csoundengine.session.SessionEvent] = []
        self._oldRenderer = None
        self._oldSessionSchedCallback = None
        self._finishedCallback = whenfinished

    def _repr_html_(self):
        if self.synthgroup is not None:
            return self.synthgroup._repr_html_()
        return repr(self)

    def schedEvent(self, event: SynthEvent) -> None:
        """
        Schedule one event to be played when we exit the context

        Args:
            event: the event to schedule

        """
        self.events.append(event)

    def schedEvents(self, events: list[SynthEvent]) -> None:
        """
        Schedule multiple events at once

        Args:
            events: the events to schedule

        """
        self.events.extend(events)

    def __enter__(self):
        """
        Performs initialization of the context

        If not called as a context manager, this method together with `exitContext`
        can be called manually to produce the same effect.

        """
        self.engine = playEngine()
        self.session = self.engine.session()

        workspace = getWorkspace()
        self._oldRenderer = workspace.renderer
        workspace.renderer = self
        self._oldSessionSchedCallback = self.session._schedCallback
        self.session._schedCallback = self.sched

        if self.lock:
            self.engine.pushLock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Executes the operations at context end

        This includes preparing all resources and then actually
        scheduling all events
        """
        if exc_type is not None:
            # There was an exception since entering
            logger.warning("Playing aborted")
            return

        if not self.events:
            logger.debug("No events scheduled, exiting context")
            self.synthgroup = None
            return

        presetToInstr = _prepareEvents(self.events, self.session, block=False)
        if self._realtimeEvents:
            for ev in self._realtimeEvents:
                self.session.prepareSched(instrname=ev.instrname,
                                          priority=ev.priority,
                                          block=False)
        self.engine.sync()
        self.session._schedCallback = self._oldSessionSchedCallback
        self._oldSessionSchedCallback = None

        synthgroup = _playFlatEvents(self.events, presetToInstr=presetToInstr,
                                     whenfinished=self._finishedCallback)

        if self._realtimeEvents:
            livesynths = [self.session.schedEvent(ev)
                          for ev in self._realtimeEvents]
            synthgroup.extend(livesynths)
        if self.lock:
            self.engine.popLock()
        self.synthgroup = synthgroup
        workspace = getWorkspace()
        workspace.renderer = self._oldRenderer
        self._oldRenderer = None
        self._oldSessionSchedCallback = None

    def sched(self,
              instrname: str,
              delay=0.,
              dur=-1.,
              priority=1,
              args: list[float] | dict[str, float] = None,
              tabargs: dict[str, float] = None,
              whenfinished=None,
              relative=None,
              **kws) -> csoundengine.session.SessionEvent:
        """
        Schedule a csound event in the active Session

        This method should be used to schedule non-preset based instruments
        when rendering in realtime (things like global effects, for example),

        Args:
            instrname: the instr. name
            delay: start time
            dur: duration
            priority: priority of the event
            args: any pfields passed to the instr., starting at p5
            tabargs: table args accepted by the instr.
            whenfinished: dummy arg, just included to keep the signature of Session.sched
            relative: the same as whenfinished: just a placeholder
            **kws: named pfields

        Returns:
            a csoundengine's SessionEvent (TODO: add link to documentation of SessionEvent)

        Example
        ~~~~~~~

        Schedule a reverb at a higher priority to affect all notes played. Notice
        that the reverb instrument is declared at the play Session (see
        :func:`playback.playSession <maelzel.core.playback.playSession>`). All instruments
        registered at this Session are immediately available for offline rendering.

        >>> from maelzel.core import *
        >>> session = playSession()
        >>> session.defInstr('reverb', r'''
        >>> |kfeedback=0.85|
        ... a1, a2 monitor
        ... aL, aR  reverbsc a1, a2, kfeedback, 12000, sr, 0.5, 1
        ... outch 1, aL - a1, 2, aR - a2
        ... ''')
        >>> chain = Chain([Note(m, 0.5) for m in range(60, 72)])
        >>> with synchedplay() as s:
        >>>     chain.play(position=1, instr='piano')
        >>>     s.sched('reverb', 0, dur=10, priority=2, args={'kfeedback':0.9})
       """
        if not instrname in self.session.instrs:
            logger.error(f"Unknown instrument {instrname}. "
                         f"Defined instruments: {self.session.registeredInstrs().keys()}")
            raise ValueError(f"Instrument {instrname} unknown")
        self.session.prepareSched(instrname, priority=priority)
        event = csoundengine.session.SessionEvent(instrname=instrname,
                                                  delay=delay,
                                                  dur=dur,
                                                  priority=priority,
                                                  args=args,
                                                  tabargs=tabargs,
                                                  kws=kws)
        self._realtimeEvents.append(event)
        return event


