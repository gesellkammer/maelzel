from __future__ import annotations

import os
import subprocess
from math import ceil

import emlib.misc
import numpy as np

import csoundengine.schedevent
import csoundengine.event
import csoundengine.sessionhandler


from maelzel.core import (
    renderer,
    synthevent,
    errors,
    playback,
    mobj,
    _playbacktools,
    )

from maelzel import _util
from maelzel.core.workspace import Workspace
from maelzel.core._common import logger

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Sequence, Callable
    import csoundengine.tableproxy
    import csoundengine.offline
    import csoundengine.instr
    import csoundengine.session

    from maelzel.snd import audiosample
    from maelzel.core.presetdef import PresetDef



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
        numchannels: number of channels of this renderer. If not set, this will
            depend on the scheduled events and the final call to .render
        verbose: if True, debugging output is show. If None, defaults to
            config (:ref:`key: 'rec.verbose' <config_rec_verbose>`)
        endtime: used when the OfflineRenderer is called as a context manager


    If rendering offline in tandem with audio samples and other csoundengine's
    functionality, it is possible to access the underlying csoundengine's OfflineSession
    via the ``.session`` attribute

    .. [1] To get the current *record path*: ``getWorkspace().recordPath()``

    (see :meth:`~maelzel.core.workspace.Workspace.recordPath`)

    .. _offlineRendererExample:

    Example
    ~~~~~~~

    Render a chromatic scale in sync with a soundfile

        >>> from maelzel.core import *
        >>> notes = [Note(n, dur=0.5) for n in range(48, 72)]
        >>> chain = Chain(notes)
        >>> defPresetSoundfont('piano', sf2path='/path/to/piano.sf2')
        >>> with render('scale.wav') as r:
        ...     chain.play(instr='piano')
        ...     # This allows access to the underlying csound offline session
        ...     r.session.playSample('/path/to/soundfile')

    When exiting the context manager the file 'scale.wav' is rendered. During
    the context manager, all calls to .play are intersected and scheduled
    via the OfflineRenderer. This makes it easy to switch between realtime
    and offline rendering by simply changing from :func:`play <maelzel.core.playback.play>`
    to :func:`render`
    """
    def __init__(self,
                 outfile='',
                 sr=0,
                 ksmps=0,
                 numchannels=0,
                 tail=0.,
                 verbose: bool | None = None,
                 endtime=0.,
                 session: csoundengine.session.Session | None = None):
        from maelzel.core import presetmanager
        super().__init__(presetManager=presetmanager.presetManager)
        w = Workspace.active
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

        self._liveSession: csoundengine.session.Session | None = session
        """A reference to the active live Session"""

        self.endtime = endtime
        """Default endtime"""

        self.tail = tail
        """Extra time at the end of rendering to make space for reverbs or long-decaying sounds"""

        self._renderProc: subprocess.Popen | None = None

        # noinspection PyUnresolvedReferences
        self._oldSessionSchedCallback: Callable | None = None
        """A reference to a schedCallback of the Session pre __enter__"""

        self._workspace: Workspace = w
        """The workspace at the moment of __enter__. Its renderer attr is modified
        and needs to be restored at __exit__"""

        self.showAtExit = False
        """Display the results at exit if running in jupyter"""

        self.session: csoundengine.offline.OfflineSession = self._makeCsoundRenderer()
        """The actual csoundengine.OfflineSession"""

    def isRealtime(self) -> bool:
        """Is this a realtime renderer?"""
        return False

    def liveSession(self) -> csoundengine.session.Session | None:
        """
        Return the realtime Session associated with this OfflineRenderer, if any
        """
        if self._liveSession is not None:
            return self._liveSession
        elif playback.isSessionActive():
            self._liveSession = playback.getSession()
            return self._liveSession
        return None

    def _makeCsoundRenderer(self) -> csoundengine.offline.OfflineSession:
        """
        Construct an :class:`csoundengine.OfflineSession` from this OfflineRenderer

        Returns:
            the corresponding :class:`csoundengine.offline.OfflineSession`
        """
        renderer = self.presetManager.makeRenderer(self.sr, ksmps=self.ksmps,
                                                   numChannels=self.numChannels)
        session = self.liveSession()
        if session:
            engine = session.engine
            for s, idx in engine.definedStrings().items():
                renderer.strSet(s, idx)
        for instr in playback._builtinInstrs():
            renderer.registerInstr(instr)
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
        if instrname not in self.session.registeredInstrs():
            self.registerInstr(name=instrname, instrdef=instr)
        self.session.commitInstrument(instrname, priority)
        return False

    def getInstr(self, instrname: str) -> csoundengine.instr.Instr | None:
        """
        Get the csoundengine's Instr corresponding to *instrname*

        Args:
            instrname: the name of the csoundengine's Instr

        Returns:
            If found, the csoundengine's Instr
        """
        instr = self.session.getInstr(instrname)
        if instr is None:
            session = playback.getSession()
            instr = session.getInstr(instrname)
            if instr is None:
                return None
            self.registerInstr(instrname, instr)
        return instr

    @property
    def scheduledEvents(self) -> dict[int, csoundengine.schedevent.SchedEvent]:
        """The scheduled events"""
        return self.session.scheduledEvents

    def assignBus(self, kind='', value=None, persist=False) -> int:
        """
        Assign a bus of the given kind

        Returns:
            the bus token. Can be used with any bus opcode (busin, busout, busmix, etc)
        """
        bus = self.session.assignBus()
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
        self.session.includeFile(path)

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
        config = Workspace.getConfig()

        from maelzel import colortheory
        blue = colortheory.safeColors['blue1']
        if not os.path.exists(sndfile):
            info = f'lastOutfile=<code style="color:{blue}">"{sndfile}"</code>'
            return f'<strong>OfflineRenderer</strong>({info})'
        from maelzel.snd import audiosample
        sample = audiosample.Sample(sndfile)
        plotHeight = config['soundfilePlotHeight']
        plotWidth = config['.soundfilePlotWidth']

        plotHeightChannel = plotHeight * (0.8 ** (sample.numchannels - 1))
        figsize = (plotWidth, plotHeightChannel * sample.numchannels)
        samplehtml = sample.reprHtml(withHeader=False, withAudiotag=True, figsize=figsize)
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
            self.session.compile(presetdef.init)
        self.registeredPresets[presetdef.name] = presetdef
        return False

    def registerInstr(self, name: str, instrdef: csoundengine.instr.Instr
                      ) -> None:
        """
        Register a csoundengine.instr.Instr to be used with this OfflineRenderer

        .. note::

            All :class:`csoundengine.instr.Instr` defined in the play Session are
            available to be rendered offline without the need to be registered

        Args:
            name: the name of this preset
            instrdef: the csoundengine.instr.Instr instance

        """
        self.instrs[name] = instrdef
        self.session.registerInstr(instrdef)

    def play(self, obj: mobj.MObj, **kws) -> csoundengine.schedevent.SchedEventGroup:
        """
        Schedule the events generated by this obj to be renderer offline

        Args:
            obj: the object to be played offline
            kws: any keyword passed to the .events method of the obj

        Returns:
            the offline score events
        """
        events = obj.synthEvents(**kws)
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
                          args=event.args,  # type: ignore
                          **kws)            # type: ignore

    def schedEvent(self, event: synthevent.SynthEvent | csoundengine.event.Event
                   ) -> csoundengine.schedevent.SchedEvent:
        """
        Schedule a SynthEvent or a csound event

        Args:
            event: a :class:`~maelzel.core.synthevent.SynthEvent`

        Returns:
            a SchedEvent

        """
        if isinstance(event, synthevent.SynthEvent):
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
            return self.session.sched(instrname=instr.name,
                                             delay=event.delay,
                                             dur=event.dur,
                                             args=pfields5,
                                             priority=event.priority,
                                             **dynargs)  # type: ignore
        elif isinstance(event, csoundengine.event.Event):
            return self._schedSessionEvent(event)
        else:
            raise TypeError(f"Expected a SynthEvent or a csound event, got {event}")

    def schedEvents(self,
                    coreevents: Sequence[synthevent.SynthEvent],
                    sessionevents: Sequence[csoundengine.event.Event] = (),
                    whenfinished: Callable | None = None
                    ) -> csoundengine.schedevent.SchedEventGroup:
        """
        Schedule multiple events as returned by :meth:`MObj.synthEvents() <maelzel.core.MObj.events>`

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
            >>> renderer.schedEvents(scale.synthEvents(instr='piano'))
            >>> renderer.render('outfile.wav')
        """
        scoreEvents = [self.schedEvent(ev) for ev in coreevents]
        if sessionevents:
            scoreEvents.extend(self._schedSessionEvent(ev) for ev in sessionevents)
        return csoundengine.schedevent.SchedEventGroup(scoreEvents)

    def definedInstrs(self) -> dict[str, csoundengine.instr.Instr]:
        """
        Get all instruments available within this OfflineRenderer

        All presets and all extra intruments registered at the active
        Session (as returned via :func:`getSession <maelzel.core.playback.getSession>`)
        are available

        Returns:
            dict `{instrname: csoundengine.instr.Instr}` with all instruments available

        """
        from maelzel.core import playback
        instrs = {}
        instrs.update(self.session.registeredInstrs())
        instrs.update(playback.getSession().registeredInstrs())
        return instrs

    def playSample(self,
                   source: int | str | tuple[np.ndarray, int] | audiosample.Sample,
                   delay=0.,
                   dur=0,
                   chan=1,
                   gain=1.,
                   speed=1.,
                   loop=False,
                   pos=0.5,
                   skip=0.,
                   fade: float | tuple[float, float] | None = None,
                   crossfade=0.02,
                   ) -> csoundengine.schedevent.SchedEvent:
        """
        Play a sample through this renderer

        Args:
            source: a soundfile, a TableProxy, a tuple (samples, sr) or a maelzel.snd.audiosample.Sample
            delay: when to play
            dur: the duration. -1 to play until the end (will detect the end of the sample)
            chan: the channel to output to
            gain: a gain applied
            speed: playback speed
            loop: should the sample be looped?
            pos: the panning position
            skip: time to skip from the audio sample
            fade: a fade applied to the playback
            crossfade: a crossfade time when looping

        Returns:
            a csoundengine.offline.SchedEvent

        """
        if not isinstance(source, (int, str, tuple)):
            from maelzel.snd import audiosample
            if isinstance(source, audiosample.Sample):
                source = (source.samples, source.sr)
            else:
                raise TypeError(f"Invalid source type: {type(source)}")
        return self.session.playSample(source=source, delay=delay, dur=dur, chan=chan,
                                              gain=gain, speed=speed, loop=loop, pan=pos,
                                              skip=skip, fade=fade, crossfade=crossfade)

    def sched(self,
              instrname: str,
              delay=0.,
              dur=-1.,
              priority=1,
              args: list[float] | dict[str, float] | None = None,
              whenfinished=None,
              relative=True,
              **kws) -> csoundengine.schedevent.SchedEvent:
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
        :func:`getSession() <maelzel.core.playback.getPlaySession>`). All instruments
        registered at this Session are immediately available for offline rendering.

            >>> from maelzel.core import *
            >>> scale = Chain([Note(n) for n in "4C 4D 4E 4F 4G".split()])
            >>> session = getSession()
            >>> session.defInstr('reverb', r'''
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
        if self.session.getInstr(instrname) is None and playback.isSessionActive():
            # Instrument not defined, try to get it from the current session
            session = playback.getSession()
            instr = session.getInstr(instrname)
            if not instr:
                logger.error(f"Unknown instrument {instrname}. "
                             f"Defined instruments: {self.session.registeredInstrs().keys()}")
                raise ValueError(f"Instrument {instrname} unknown")
            self.session.registerInstr(instr)
        return self.session.sched(instrname=instrname,
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
               endtime: float | None = None,
               ksmps: int | None = None,
               tail: float| None  = None,
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
            >>> renderer.schedEvents(scale.synthEvents(instr='piano'))
            >>> renderer._sched('reverb', priority=2)
            >>> renderer.render('outfile.wav')

        """
        self._renderProc = None
        cfg = Workspace.active.config
        if outfile == '?':
            from maelzel.core import _dialogs
            outfile = _dialogs.saveRecordingDialog()
            if not outfile:
                raise errors.CancelledError("Render operation was cancelled")
        elif not outfile:
            outfile = self._outfile or _playbacktools.makeRecordingFilename(ext=".wav")
        outfile = _util.normalizeFilename(outfile)
        if verbose is None:
            verbose = self._verbose if self._verbose is not None else cfg['rec.verbose']
        job = self.session.render(outfile=outfile,
                                  ksmps=ksmps or self.ksmps,
                                  wait=wait if wait is not None else cfg['rec.blocking'],
                                  verbose=verbose,
                                  openWhenDone=openWhenDone,
                                  compressionBitrate=compressionBitrate or cfg['.rec.compressionBitrate'],
                                  endtime=endtime if endtime is not None else self.endtime,
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
        lastjob = self.session.renderedJobs[-1] if self.session.renderedJobs else None
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
        return self.session.generateCsdString()

    def writeCsd(self, outfile='?') -> str:
        """
        Write the .csd which would render all events scheduled until now

        Args:
            outfile: the path of the saved .csd

        Returns:
            the outfile
        """
        if outfile == "?":
            from maelzel.core import _dialogs
            selected = _dialogs.selectFileForSave("saveCsdLastDir", filter="Csd (*.csd)")
            if not selected:
                raise errors.CancelledError("Save operation cancelled")
            outfile = selected
        self.session.writeCsd(outfile)
        return outfile

    def getSynth(self, token: int) -> csoundengine.schedevent.SchedEvent | None:
        return self.session.getEventById(token)

    def __enter__(self):
        """
        When used as a context manager, every call to .play will be diverted to be
        recorded offline

        """
        self._workspace = Workspace.active
        self._oldRenderer = self._workspace.renderer
        self._workspace.renderer = self
        session = self.liveSession()
        if session:
            session.setHandler(_OfflineSessionHandler(self))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            # There was an exception since entering
            logger.warning("Offline rendering aborted")
            return
        self._workspace.renderer = self._oldRenderer
        self._workspace = Workspace.active
        self._oldRenderer = None

        session = self.liveSession()
        if session:
            session.setHandler(None)
            # self._session.setSchedCallback(self._oldSessionSchedCallback)
            # self._oldSessionSchedCallback = None
            # self._session = None

        outfile = self._outfile or _playbacktools.makeRecordingFilename()
        logger.info(f"Rendering to {outfile}")
        self.render(outfile=outfile, wait=True)
        if self.showAtExit:
            self.show()

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

    def readSoundfile(self, soundfile: str, chan=0, skiptime=0.
                      ) -> csoundengine.tableproxy.TableProxy:
        return self.session.readSoundfile(path=soundfile, chan=chan, skiptime=skiptime)

    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int | tuple[int, int] = 0,
                  sr: int = 0,
                  tabnum: int = 0
                  ) -> csoundengine.tableproxy.TableProxy:
        """
        Create a table in this renderer

        Args:
            data: if given, the table will be created with the given data
            size: if data is not given, an empty table of the given size is created. Otherwise,
                this parameter is ignored. A multichannel table can be created by specifying
                the size as a tuple ``(numframes: int, numchannels: int)``
            sr: the sample rate of the data, if applicable
            tabnum: leave it as 0 to let the renderer assign a table number

        Returns:
            the assigned table number
        """
        if (data is not None and size) or (data is None and not size):
            raise ValueError("Either data or size must be given, not both")
        return self.session.makeTable(data=data, size=size, sr=sr, tabnum=tabnum)

    def wait(self, timeout=0) -> None:
        """
        Wait until finished rendering

        Args:
            timeout: a timeout (0 to wait indefinitely)

        """
        proc = self.lastRenderProc()
        if proc is not None and proc.poll() is None:
            proc.wait(timeout=timeout)


class _OfflineSessionHandler(csoundengine.sessionhandler.SessionHandler):
    def __init__(self, renderer: OfflineRenderer):
        self.renderer = renderer

    def sched(self, event: csoundengine.event.Event):
        return self.renderer._schedSessionEvent(event)

    def schedEvent(self, event: csoundengine.event.Event) -> csoundengine.schedevent.SchedEvent:
        return self.renderer.schedEvent(event)

    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int | tuple[int, int] = 0,
                  sr: int = 0,
                  ) -> csoundengine.tableproxy.TableProxy:
        return self.renderer.makeTable(data=data, size=size, sr=sr)

    def readSoundfile(self,
                      path: str,
                      chan=0,
                      skiptime=0.,
                      delay=0.,
                      force=False,
                      ) -> csoundengine.tableproxy.TableProxy:
        return self.renderer.readSoundfile(soundfile=path, chan=chan, skiptime=skiptime)


def render(outfile='',
           events: Sequence[synthevent.SynthEvent | mobj.MObj | csoundengine.event.Event | Sequence[mobj.MObj | synthevent.SynthEvent]] = (),
           sr: int = 0,
           wait: bool | None = None,
           ksmps=0,
           verbose: bool | None = None,
           nchnls: int | None = None,
           workspace: Workspace | None = None,
           tail: float | None = None,
           run=True,
           endtime=0.,
           show=False,
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
        endtime: if given, sets the end time of the rendered segment. A value
            of 0. indicates to render everything. A value is needed if there
            are endless events
        show: display the resulting OfflineRenderer when running inside jupyter
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
        ...     a.synthEvents(chain=1),
        ...     b.synthEvents(chan=2, gain=0.2)
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
        cfg = Workspace.active.config
        tail = cfg['rec.extratime']
    assert isinstance(tail, (int, float))

    if not events:
        # called as a context manager
        offlinerenderer = OfflineRenderer(outfile=outfile,
                                          sr=sr,
                                          numchannels=nchnls or 0,
                                          verbose=verbose,
                                          ksmps=ksmps,
                                          tail=tail,
                                          endtime=endtime)
        return offlinerenderer
    if workspace is None:
        workspace = Workspace.active
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
