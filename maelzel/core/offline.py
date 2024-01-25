from __future__ import annotations

import os
import subprocess
from math import ceil

import csoundengine
import emlib.misc
import numpy as np

from maelzel.core import renderer
from maelzel.core import presetmanager
from maelzel.core.workspace import Workspace
from maelzel.core.presetdef import PresetDef
from maelzel.core.synthevent import SynthEvent
from maelzel.core.errors import CancelledError
from maelzel.core import playback
from maelzel.core import mobj
from maelzel.core._common import logger
from maelzel.core import _dialogs
from maelzel.core import _playbacktools
from maelzel import _util
from maelzel.snd import audiosample

from typing import Callable, Sequence


__all__ = (
    'render',
    'OfflineRenderer'
)

# -------------------------------------------------------------
#                        OfflineRenderer
# -------------------------------------------------------------


class OfflineRenderer(renderer.Renderer):
    """
    An OfflineRenderer is created to render musical objects to a soundfile

    .. admonition:: OfflineRenderer as context manager

        The simplest way to render offline is to use an OfflineRenderer
        as a context manager (see also :func:`render`). Within this
        context any .play call will be collected and everything will be
        rendered when exiting the context
        (:ref:`see example below <offlineRendererExample>`)

    Args:
        outfile: the path to the rendered soundfile. If not given, a path
            within the record path [1]_ is returned
        sr: the sr of the render (:ref:`config key: 'rec.sr' <config_rec_sr>`)
        ksmps: the ksmps used for the recording
        verbose: if True, debugging output is show. If None, defaults to
            config (:ref:`key: 'rec.verbose' <config_rec_verbose>`)

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
        ...     # This allows access to the underlying csound renderer
        ...     r.renderer.playSample((samples, sr))

    When exiting the context manager the file 'scale.wav' is rendered. During
    the context manager, all calls to .play are intersected and scheduled
    via the OfflineRenderer. This makes it easy to switch between realtime
    and offline rendering by simply changing from :func:`play <maelzel.core.playback.play>`
    to :func:`render`
    """
    def __init__(self,
                 outfile: str = '',
                 sr: int | None = None,
                 ksmps: int | None = None,
                 numchannels: int | None = None,
                 tail=0.,
                 verbose: bool = None):

        super().__init__(presetManager=presetmanager.presetManager)
        w = Workspace.getActive()
        cfg = w.config

        self._outfile = outfile
        """Outfile given for rendering"""

        self.a4 = w.a4
        """A value for the reference frequency"""

        self.sr = sr or cfg['rec.sr']
        """The sr. If not given, ['rec.sr'] is used """

        self.ksmps = ksmps or cfg['rec.ksmps']
        """ksmps value (samples per block)"""

        self.numChannels = numchannels or cfg['rec.numChannels']

        self.instrs: dict[str, csoundengine.instr.Instr] = {}
        """An index of registered Instrs, mapping name to the Instr instance"""

        self.renderedSoundfiles: list[str] = []
        """A list of soundfiles rendered with this renderer"""

        self._verbose = verbose

        self._renderProc: subprocess.Popen | None = None

        self.csoundRenderer: csoundengine.offline.Renderer = self.makeRenderer()
        """The actual csoundengine.Renderer"""

        self.tail = tail
        """Extra time at the end of rendering to make space for reverbs or long-decaying sounds"""

        self._session: csoundengine.session.Session | None = None
        """A reference to the playback Session"""

        # noinspection PyUnresolvedReferences
        self._oldSessionSchedCallback: Callable | None = None
        """A reference to a schedCallback of the Session pre __enter__"""

        self._workspace: Workspace = w
        """The workspace at the moment of __enter__. Its renderer attr is modified
        and needs to be restored at __exit__"""

    def isRealtime(self) -> bool:
        """Is this a realtime renderer?"""
        return False

    def makeRenderer(self) -> csoundengine.offline.Renderer:
        """
        Construct a :class:`csoundengine.Renderer` from this OfflineRenderer

        Returns:
            the corresponding :class:`csoundengine.Renderer`
        """
        from maelzel.core import playback
        renderer = self.presetManager.makeRenderer(self.sr, ksmps=self.ksmps,
                                                   numChannels=self.numChannels)
        if playback.isSessionActive():
            engine = playback._playEngine()
            for s, idx in engine.definedStrings().items():
                renderer.strSet(s, idx)
        return renderer

    def prepareInstr(self, instr: csoundengine.instr.Instr, priority: int
                     ) -> bool:
        """
        Reify an instance of *instr* at the given priority

        This method also prepares any resources and initialization that the given
        Instr might have

        Args:
            instr: a csoundengine's Instr
            priority: the priority to instantiate this instr with. Priorities
                start with 1

        Returns:
            False

        """
        instrname = instr.name
        if instrname not in self.csoundRenderer.registeredInstrs():
            self.registerInstr(name=instrname, instrdef=instr)
        self.csoundRenderer.commitInstrument(instrname, priority)
        return False

    def getInstr(self, instrname: str) -> csoundengine.instr.Instr | None:
        """
        Get the csoundengine's Instr corresponding to *instrname*

        Args:
            instrname: the name of the csoundengine's Instr

        Returns:
            If found, the csoundengine's Instr
        """
        instr = self.csoundRenderer.getInstr(instrname)
        if instr is None:
            session = playback.playSession()
            instr = session.getInstr(instrname)
            if instr is None:
                return None
            self.registerInstr(instrname, instr)
        return instr

    @property
    def scheduledEvents(self) -> dict[int, csoundengine.offline.SchedEvent]:
        """The scheduled events"""
        return self.csoundRenderer.scheduledEvents

    def assignBus(self, kind='', value=None, persist=False) -> int:
        """
        Assign a bus of the given kind

        Returns:
            the bus token. Can be used with any bus opcode (busin, busout, busmix, etc)
        """
        bus = self.csoundRenderer.assignBus()
        return bus.token

    def releaseBus(self, busnum: int) -> None:
        """
        Signal that we no longer use the given bus

        Args:
            busnum: the bus token as returned by :meth:`OfflineRenderer.assignBus`

        """
        pass

    def includeFile(self, path: str) -> None:
        """
        Add an include clause to this renderer.

        OfflineRenderer keeps track of includes so trying to include the same file
        multiple times will generate only one #include clause

        Args:
            path: the path of the file to include

        """
        self.csoundRenderer.includeFile(path)

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
        header = '<strong>OfflineRenderer</strong>'

        def _(s):
            return f'<code style="color:{blue}">{s}</code>'

        sndfilestr = f'"{sndfile}"'
        info = f'outfile={_(sndfilestr)}, {_(sample.numchannels)} channels, ' \
               f'{_(format(sample.duration, ".2f"))} secs, {_(sample.sr)} Hz'
        header = f'{header}({info})'
        return '<br>'.join([header, samplehtml])

    def registerPreset(self, presetdef: PresetDef) -> bool:
        """
        Register the given PresetDef with this renderer

        Args:
            presetdef: the preset to register. Any global/init code declared by
                the preset will be made available to this renderer

        Returns:
            to adjust to the Renderer parent class we always return
            False since offline rendering does not need to sync

        """
        if presetdef.name in self.registeredPresets:
            return False
        instr = presetdef.getInstr()
        self.registerInstr(instr.name, instr)
        if presetdef.includes:
            for include in presetdef.includes:
                self.includeFile(include)
        if presetdef.init:
            self.csoundRenderer.addGlobalCode(presetdef.init)
        self.registeredPresets[presetdef.name] = presetdef
        return False

    def registerInstr(self, name: str, instrdef: csoundengine.instr.Instr
                      ) -> None:
        """
        Register a csoundengine.Instr to be used with this OfflineRenderer

        .. note::

            All :class:`csoundengine.Instr` defined in the play Session are
            available to be rendered offline without the need to be registered

        Args:
            name: the name of this preset
            instrdef: the csoundengine.Instr instance

        """
        self.instrs[name] = instrdef
        self.csoundRenderer.registerInstr(instrdef)

    def play(self, obj: mobj.MObj, **kws) -> csoundengine.offline.SchedEventGroup:
        """
        Schedule the events generated by this obj to be renderer offline

        Args:
            obj: the object to be played offline
            kws: any keyword passed to the .events method of the obj

        Returns:
            the offline score events
        """
        events = obj.events(**kws)
        return self.schedEvents(events)

    def prepareSessionEvent(self, sessionevent: csoundengine.event.Event
                            ) -> None:
        """
        Prepare a session event

        Args:
            sessionevent: the session event to prepare. This is mostly used internally

        """
        pass

    def _schedSessionEvent(self, event: csoundengine.event.Event
                           ) -> csoundengine.schedevent.SchedEvent:
        """
        Schedule a Session event at this renderer

        Args:
            event: the event to schedule

        Returns:
            a ScoreEvent corresponding to keep track of the scheduled event

        .. seealso:: https://csoundengine.readthedocs.io/en/latest/api/csoundengine.offline.ScoreEvent.html#csoundengine.offline.ScoreEvent

        """
        kws = event.kws if event.kws is not None else {}
        return self.sched(instrname=event.instrname,
                          delay=event.delay,
                          dur=event.dur,
                          priority=event.priority,
                          args=event.args,
                          **kws)

    def schedEvent(self, event: SynthEvent | csoundengine.event.Event
                   ) -> csoundengine.schedevent.SchedEvent:
        """
        Schedule a SynthEvent or a csound event

        Args:
            event: a :class:`~maelzel.core.synthevent.SynthEvent`

        Returns:
            a SchedEvent

        """
        if isinstance(event, csoundengine.event.Event):
            return self._schedSessionEvent(event)
        elif isinstance(event, SynthEvent):
            if event.initfunc:
                event.initfunc(event, self)
            presetname = event.instr
            instr = self.instrs.get(presetname)
            if instr is None:
                preset = self.presetManager.getPreset(presetname)
                if not preset:
                    raise ValueError(f"Unknown preset instr: {presetname}")
                self.preparePreset(preset, event.priority)
                instr = preset.getInstr()
            pfields5, dynargs = event._resolveParams(instr)
            return self.csoundRenderer.sched(instrname=instr.name,
                                             delay=event.delay,
                                             dur=event.dur,
                                             args=pfields5,
                                             priority=event.priority,
                                             **dynargs)
        else:
            raise TypeError(f"Expected a SynthEvent or a csound event, got {event}")

    def schedEvents(self,
                    coreevents: list[SynthEvent],
                    sessionevents: list[csoundengine.event.Event] = None,
                    whenfinished: Callable = None
                    ) -> csoundengine.offline.SchedEventGroup:
        """
        Schedule multiple events as returned by :meth:`MObj.events() <maelzel.core.MObj.events>`

        Args:
            coreevents: the events to schedule
            sessionevents: csound events as packed within a csoundengine.session.SessionEvent
            whenfinished: dummy arg, here to conform to the signature of the parent. Only makes
                sense in realtime

        Returns:
            a :class:`csoundengine.offline.SchedEventGroup`. This can be used to modify
            scheduled events via :meth:`set`, :meth:`automate` or `stop`

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> scale = Chain([Note(m, 0.5) for m in range(60, 72)])
            >>> renderer = OfflineRenderer()
            >>> renderer.schedEvents(scale.events(instr='piano'))
            >>> renderer.render('outfile.wav')
        """
        scoreEvents = [self.schedEvent(ev) for ev in coreevents]
        if sessionevents:
            scoreEvents.extend(self._schedSessionEvent(ev) for ev in sessionevents)
        return csoundengine.offline.SchedEventGroup(scoreEvents)

    def definedInstrs(self) -> dict[str, csoundengine.instr.Instr]:
        """
        Get all instruments available within this OfflineRenderer

        All presets and all extra intruments registered at the active
        Session (as returned via getPlaySession) are available

        Returns:
            dict {instrname: csoundengine.Instr} with all instruments available

        """
        from maelzel.core import playback
        instrs = {}
        instrs.update(self.csoundRenderer.registeredInstrs())
        instrs.update(playback.playSession().registeredInstrs())
        return instrs

    def sched(self,
              instrname: str,
              delay=0.,
              dur=-1.,
              priority=1,
              args: list[float] | dict[str, float] = None,
              whenfinished=None,
              relative=True,
              **kws) -> csoundengine.offline.SchedEvent:
        """
        Schedule a csound event

        This method can be used to schedule non-preset based instruments
        when rendering offline (things like global effects, for example),
        similarly to how a user might schedule a non-preset based instrument
        in real-time.

        If an OfflineRenderer is used as a context manager it is also possible
        to call the session's ._sched method directly since its _sched callback is
        rerouted to call this OfflineRenderer instead

        Args:
            instrname: the instr. name
            delay: start time
            dur: duration
            priority: priority of the event
            args: any pfields passed to the instr., starting at p5
            whenfinished: this argument does nothing under this context. It is only
                present to make the signature compatible with the interface
            relative: dummy argument, here to conform to the signature of
                csoundengine's Session.sched, which is redirected to this
                method when an OfflineRenderer is used as a context manager
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
            ...     r._sched('reverb', priority=2)
            ...     scale.play('piano')

        """
        if self.csoundRenderer.getInstr(instrname) is None and playback.isSessionActive():
            # Instrument not defined, try to get it from the current session
            session = playback.playSession()
            instr = session.getInstr(instrname)
            if not instr:
                logger.error(f"Unknown instrument {instrname}. "
                             f"Defined instruments: {self.csoundRenderer.registeredInstrs().keys()}")
                raise ValueError(f"Instrument {instrname} unknown")
            self.csoundRenderer.registerInstr(instr)
        return self.csoundRenderer.sched(instrname=instrname,
                                         delay=delay,
                                         dur=dur,
                                         priority=priority,
                                         args=args,
                                         **kws)

    def render(self,
               outfile='',
               wait: bool | None = None,
               verbose: bool | None = None,
               openWhenDone=False,
               compressionBitrate: int | None = None,
               endtime=0.,
               ksmps: int | None = None,
               tail: float = None,
               ) -> str:
        """
        Render the events scheduled until now.

        You can access the rendering subprocess (a :class:`subprocess.Popen` object)
        via :meth:`~OfflineRenderer.lastRenderProc`

        Args:
            outfile: the soundfile to generate. Use "?" to save via a GUI dialog,
                None will render to a temporary file
            wait: if True, wait until rendering is done
            verbose: if True, show output generated by csound itself
                (print statements and similar opcodes still produce output)
            endtime: if given, crop rendering to this absolute time (in seconds)
            compressionBitrate: the compression bit rate when rendering to .ogg
                (in kb/s, the default can be configured in
                `config['.rec.compressionBitrate'] <_config_rec_compressionbitrate>`
            openWhenDone: if True, open the rendered soundfile in the default
                application
            ksmps: the samples per cycle used when rendering
            tail: an extra time at the end of the render to make room for long decaying
                sounds / reverbs. If given, overrides the tail parameter given at init.

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
            >>> renderer._sched('reverb', priority=2)
            >>> renderer.render('outfile.wav')

        """
        self._renderProc = None
        cfg = Workspace.getActive().config
        if outfile == '?':
            outfile = _dialogs.saveRecordingDialog()
            if not outfile:
                raise CancelledError("Render operation was cancelled")
        elif not outfile:
            outfile = self._outfile or _playbacktools.makeRecordingFilename(ext=".wav")
        outfile = _util.normalizeFilename(outfile)
        if verbose is None:
            verbose = self._verbose if self._verbose is not None else cfg['rec.verbose']
        job = self.csoundRenderer.render(outfile=outfile,
                                         ksmps=ksmps or self.ksmps,
                                         wait=wait if wait is not None else cfg['rec.blocking'],
                                         verbose=verbose,
                                         openWhenDone=openWhenDone,
                                         compressionBitrate=compressionBitrate or cfg['.rec.compressionBitrate'],
                                         endtime=endtime,
                                         tail=tail if tail is not None else self.tail)
        self.renderedSoundfiles.append(outfile)
        self._renderProc = job.process
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
        lastjob = self.csoundRenderer.renderedJobs[-1] if self.csoundRenderer.renderedJobs else None
        if not lastjob:
            return ''
        lastjob.wait(timeout=timeout)
        emlib.misc.open_with_app(lastjob.outfile)
        return lastjob.outfile

    def lastOutfile(self) -> str | None:
        """
        Last rendered outfile, None if no soundfiles were rendered

        Example
        ~~~~~~~

            >>> r = OfflineRenderer(...)
            >>> r._sched(...)
            >>> r.render(wait=True)
            >>> r.lastOutfile()
            '~/.local/share/maelzel/recordings/tmpsasjdas.wav'
        """
        return self.renderedSoundfiles[-1] if self.renderedSoundfiles else None

    def lastRenderProc(self) -> subprocess.Popen | None:
        """
        Last process (subprocess.Popen) used for rendering

        Example
        ~~~~~~~

            >>> r = OfflineRenderer(...)
            >>> r._sched(...)
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
        return self.csoundRenderer.generateCsdString()

    def writeCsd(self, outfile: str = '?') -> str:
        """
        Write the .csd which would render all events scheduled until now

        Args:
            outfile: the path of the saved .csd

        Returns:
            the outfile
        """
        if outfile == "?":
            selected = _dialogs.selectFileForSave("saveCsdLastDir", filter="Csd (*.csd)")
            if not selected:
                raise CancelledError("Save operation cancelled")
            outfile = selected
        self.csoundRenderer.writeCsd(outfile)
        return outfile

    def getSynth(self, token: int) -> csoundengine.schedevent.SchedEvent | None:
        return self.csoundRenderer.getEventById(token)

    def __enter__(self):
        """
        When used as a context manager, every call to .play will be diverted to be
        recorded offline

        """
        self._workspace = Workspace.getActive()
        self._oldRenderer = self._workspace.renderer
        self._workspace.renderer = self

        if playback.isSessionActive():
            self._session = session = playback.playSession()
            self._oldSessionSchedCallback = session.setSchedCallback(self._schedSessionEvent)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            # There was an exception since entering
            logger.warning("Offline rendering aborted")
            return
        self._workspace.renderer = self._oldRenderer
        self._workspace = Workspace.active
        self._oldRenderer = None

        if self._session:
            self._session.setSchedCallback(self._oldSessionSchedCallback)
            self._oldSessionSchedCallback = None
            self._session = None

        outfile = self._outfile or _playbacktools.makeRecordingFilename()
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
        return proc is not None and proc.poll() is not None

    def readSoundfile(self, soundfile: str, chan=0, skiptime=0.) -> int:
        tabproxy = self.csoundRenderer.readSoundfile(path=soundfile, chan=chan,
                                                     skiptime=skiptime)
        return tabproxy.tabnum

    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int = 0,
                  sr: int = 0,
                  tabnum: int = 0
                  ) -> int:
        """
        Create a table in this renderer

        Args:
            data: if given, the table will be created with the given data
            size: if data is not given, an empty table of the given size is created. Otherwise,
                this parameter is ignored
            sr: the sample rate of the data, if applicable
            tabnum: leave it as 0 to let the renderer assign a table number

        Returns:
            the assigned table number
        """
        if data is not None and size > 0:
            raise ValueError("Either data or size must be given, not both")
        if data is None and not size:
            raise ValueError("Either data or size must be given")
        tabproxy = self.csoundRenderer.makeTable(data=data, size=size, sr=sr, tabnum=tabnum)
        return tabproxy.tabnum

    def wait(self, timeout=0) -> None:
        """
        Wait until finished rendering

        Args:
            timeout: a timeout (0 to wait indefinitely)

        """
        proc = self.lastRenderProc()
        if proc is not None and proc.poll() is None:
            proc.wait(timeout=timeout)


def render(outfile='',
           events: Sequence[SynthEvent | mobj.MObj | csoundengine.event.Event | Sequence[mobj.MObj | SynthEvent]] = None,
           sr: int = None,
           wait: bool = None,
           ksmps: int = None,
           verbose: bool = None,
           nchnls: int = None,
           workspace: Workspace = None,
           tail: float | None = None,
           run=True,
           **kws
           ) -> OfflineRenderer:
    """
    Render to a soundfile / creates a **context manager** to render offline

    When not used as a context manager the events / objects must be given. The soundfile
    will be generated immediately.

    When used as a context manager the `events` argument should be left unset.
    Within this context any call to :meth:`maelzel.core.MObj.play` will be redirected to
    the offline renderer and at the exit of the context all events will be rendered to a
    soundfile. Also, any pure csound events scheduled via ``playSession()._sched(...)``
    will also be redirected to be renderer offline.

    This enables to use the exact same code when doing realtime and offline rendering.

    Args:
        outfile: the generated file. If None, a file inside the recording
            path is created (see `recordPath`). Use "?" to save via a GUI dialog or
        events: the events/objects to play. This can only be left unset if using ``render``
            as a context manager (see example).
        sr: sample rate of the soundfile (:ref:`config 'rec.sr' <config_rec_sr>`)
        ksmps: number of samples per cycle (:ref:`config 'rec.ksmps' <config_rec_ksmps>`)
        nchnls: number of channels of the rendered soundfile
        wait: if True, wait until recording is finished. If None,
            use the :ref:`config 'rec.blocking' <config_rec_blocking>`
        verbose: if True, show the output generated by the csound subprocess
        tail: extra time added at the end of the render, usefull when rendering reverbs or
            long decaying sound. If None, uses use :ref:`config 'rec.extratime' <config_rec_extratime>`
        run: if True, perform the render itself
        tail: extra time at the end, usefull when rendering reverbs or long deaying sounds
        workspace: if given, this workspace overrides the active workspace

    Returns:
        the :class:`OfflineRenderer` used to render the events. If the outfile
        was not given, the path of the recording can be retrieved from
        ``renderer.outfile``

    Example
    ~~~~~~~

        >>> a = Chord("A4 C5", start=1, dur=2)
        >>> b = Note("G#4", dur=4)
        >>> render("out.wav", events=[
        ...     a.events(chain=1),
        ...     b.events(chan=2, gain=0.2)
        ... ])

    This function can be also used as a context manager, similar to
    :func:`maelzel.playback.play`. In that case `events` must be ``None``:

        >>> from maelzel.core import *
        >>> scale = Chain([Note(n) for n in "4C 4D 4E 4F 4G".split()])
        >>> playSession().defInstr('reverb', r'''
        ... |kfeedback=0.6|
        ... amon1, amon2 monitor
        ... a1, a2 reverbsc amon1, amon2, kfeedback, 12000, sr, 0.6
        ... outch 1, a1-amon1, 2, a2-amon2
        ... ''')
        >>> with render() as r:
        ...     scale.play('.piano')   # .play here is redirected to the offline renderer
        ...     r.sched('reverb', priority=2)


    .. seealso:: :class:`OfflineRenderer`, :func:`maelzel.playback.play`
    """
    if tail is None:
        cfg = Workspace.getActive().config
        tail = cfg['rec.extratime']
    assert isinstance(tail, (int, float))

    if not events:
        # called as a context manager
        return OfflineRenderer(outfile=outfile,
                               sr=sr,
                               numchannels=nchnls,
                               verbose=verbose,
                               ksmps=ksmps,
                               tail=tail)
    if workspace is None:
        workspace = Workspace.getActive()
    coreEvents, sessionEvents = _playbacktools.collectEvents(events, eventparams=kws, workspace=workspace)
    if not nchnls:
        nchnls = max(int(ceil(ev.resolvedPosition() + ev.chan)) for ev in coreEvents)
    renderer = OfflineRenderer(sr=sr, ksmps=ksmps, numchannels=nchnls, tail=tail)
    if coreEvents:
        renderer.schedEvents(coreEvents)

    if sessionEvents:
        for sessionevent in sessionEvents:
            renderer._schedSessionEvent(sessionevent)

    if run:
        renderer.render(outfile=outfile, wait=wait, verbose=verbose)
    return renderer
