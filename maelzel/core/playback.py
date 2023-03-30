"""
This module handles playing of events

"""
from __future__ import annotations
import os

from datetime import datetime

import emlib.misc
import emlib.iterlib
import numpy as np

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
from .renderer import Renderer

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, Callable
    from .mobj import MObj
    from maelzel.snd import audiosample
    import subprocess


__all__ = (
    'render',
    'Synched',
    'play',
    'testAudio',
    'playEngine',
    'playSession',
    'RealtimeRenderer',
    'OfflineRenderer'
)


class RealtimeRenderer(Renderer):
    """
    A RealtimeRenderer is created whenever realtime playback is initiated.

    Normally a user does not create a RealtimeRenderer. It is created during
    playback. There are three gateways to initiate a playback process:

    * :meth:`maelzel.core.MObj.play`. The play method of an object
      (a :class:`~maelzel.core.event.Note`, a :class:`~maelzel.core.chain.Chain`, a
      :class:`~maelzel.core.score.Score`)
      is called, this creates a :class:`RealtimeRenderer` which immediately calls its
      :meth:`RealtimeRenderer.schedEvents` method
    * :func:`maelzel.core.playback.play`. This initiates playback for multiple
      objects / events and synchs the playback as if all the objects where part
      of a group.
    * :func:`maelzel.core.playback.synchedplay`. This context manager acts very similar
      to the `play` function, ensureing that playback is synched.

    """

    def __init__(self, engine: csoundengine.Engine = None):
        super().__init__(presetManager=presetManager)
        if engine is None:
            engine = playEngine()
        self.engine: csoundengine.Engine = engine
        self.session: csoundengine.Session = engine.session()

    def isRealtime(self) -> bool:
        return True

    def assignBus(self, kind='audio') -> int:
        return self.engine.assignBus(kind=kind, persist=True)

    def releaseBus(self, busnum: int):
        self.engine.releaseBus(busnum)

    def registerPreset(self, presetdef: PresetDef) -> None:
        instr = presetdef.getInstr()
        # The Session itself caches instrs and checks definitions
        isnew = self.session.registerInstr(instr)
        if isnew:
            logger.debug("*********** Session registered new instr: ", instr.name)
        self.registeredPresets[presetdef.name] = presetdef

    def getInstr(self, instrname: str) -> csoundengine.instr.Instr | None:
        return self.session.getInstr(instrname)

    def prepareInstr(self, instr: csoundengine.instr.Instr, priority: int):
        instrname = instr.name
        # TODO
        self.session.prepareSched(instrname, priority=priority)

    def prepareSessionEvent(self, sessionevent: csoundengine.session.SessionEvent):
        self.session.prepareSched(instrname=sessionevent.instrname,
                                  priority=sessionevent.priority)

    def schedSessionEvent(self, event: csoundengine.session.SessionEvent
                          ) -> csoundengine.synth.Synth:
        assert event.instrname in self.session.instrs
        kws = event.kws if event.kws is not None else {}
        return self.session.sched(instrname=event.instrname,
                                  delay=event.delay,
                                  dur=event.dur,
                                  priority=event.priority,
                                  args=event.args,
                                  tabargs=event.tabargs,
                                  **kws)

    def schedEvent(self, event: SynthEvent) -> csoundengine.synth.Synth:
        group = self.schedEvents([event])
        return group[0]

    def schedEvents(self,
                    coreevents: list[SynthEvent],
                    sessionevents: list[csoundengine.session.SessionEvent] = None,
                    whenfinished: Callable = None
                    ) -> csoundengine.synth.SynthGroup:
        synths = _schedFlatEvents(self, coreevents=coreevents, sessionevents=sessionevents,
                                  whenfinished=whenfinished)
        return csoundengine.synth.SynthGroup(synths)


    def includeFile(self, path: str) -> None:
        self.engine.includeFile(path)

    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int = 0,
                  sr: int = 0,
                  tabnum: int = 0
                  ) -> int:
        table = self.session.makeTable(data=data, size=size, sr=sr, tabnum=tabnum)
        return table.tabnum

    def sched(self,
             instrname: str,
             delay: float = 0.,
             dur: float = -1,
             args: list[float|str] | None = None,
             priority: int = 1,
             whenfinished: Callable = None):
        return self.session.sched(instrname=instrname,
                                  delay=delay,
                                  dur=dur,
                                  args=args,
                                  priority=priority,
                                  whenfinished=whenfinished)

    def pushLock(self):
        """Lock the sound engine's clock, to schedule a number of synths in sync.

        This is mostly used internally
        """
        self.engine.sync()
        self.engine.pushLock()

    def popLock(self):
        """Pop a previously pushed clock lock

        This method is mostly used internally
        """
        self.engine.popLock()


class OfflineRenderer(Renderer):
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

        super().__init__(presetManager=presetManager)
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

        self.numChannels = numchannels

        self.instrDefs: dict[str, csoundengine.Instr] = {}
        """An index of registered Instrs"""

        self.renderedSoundfiles: list[str] = []
        """A list of soundfiles rendered with this renderer"""

        self._quiet = quiet

        self._renderProc: subprocess.Popen|None = None

        self.csoundRenderer: csoundengine.Renderer = self.makeRenderer()
        """The actual csoundengine.Renderer"""

        self._session: csoundengine.session.Session | None = None
        """A reference to the playback Session"""

        self._oldSessionSchedCallback: Callable | None = None
        """A reference to a schedCallback of the Session pre __enter__"""

        self._workspace: Workspace | None = None
        """The workspace at the moment of __enter__. Its renderer attr is modified
        and needs to be restored at __exit__"""

    def isRealtime(self) -> bool:
        """Is this a realtime renderer?"""
        return False

    def makeRenderer(self) -> csoundengine.Renderer:
        """
        Construct a :class:`csoundengine.Renderer` from this OfflineRenderer

        Returns:
            the corresponding :class:`csoundengine.Renderer`
        """
        renderer = presetManager.makeRenderer(self.sr, ksmps=self.ksmps,
                                              numChannels=self.numChannels)
        if isEngineActive():
            engine = playEngine()
            for s, idx in engine.definedStrings().items():
                renderer.strSet(s, idx)
        return renderer

    def prepareInstr(self, instr: csoundengine.instr.Instr, priority: int) -> int:
        """
        Reify an instance of *instr* at the given priority

        This method also prepares any resources and initialization that the given
        Instr might have

        Args:
            instr: a csoundengine's Instr
            priority: the priority to instantiate this instr with. Priorities
                start with 1

        Returns:

        """
        instrname = instr.name
        assert self.csoundRenderer.isInstrDefined(instrname)
        return self.csoundRenderer.commitInstrument(instrname, priority)

    def getInstr(self, instrname: str) -> csoundengine.instr.Instr | None:
        """
        Get the csoundengine's Instr corresponding to *instrname*

        Args:
            instrname: the name of the csoundengine's Instr

        Returns:
            If found, the csoundengine's Instr
        """
        return self.csoundRenderer.getInstr(instrname)

    @property
    def scheduledEvents(self) -> dict[int, csoundengine.offline.ScoreEvent]:
        """The scheduled events"""
        return self.csoundRenderer.scheduledEvents

    def assignBus(self, kind='audio') -> int:
        """
        Assign a bus of the given kind

        Returns:
            the bus token. Can be used with any bus opcode (busin, busout, busmux, etc)
        """
        return self.csoundRenderer.assignBus()

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
        self.csoundRenderer.addInclude(path)

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

    def show(self) -> None:
        """
        If inside jupyter, force a display of this OfflineRenderer

        """
        if emlib.misc.inside_jupyter():
            from IPython.display import display
            display(self)

    def registerPreset(self, presetdef: PresetDef) -> None:
        """
        Register the given PresetDef with this renderer

        Args:
            presetdef: the preset to register. Any global/init code declared by
                the preset will be made available to this renderer

        """
        if presetdef.name  in self.registeredPresets:
            return
        instr = presetdef.getInstr()
        self.registerInstr(instr.name, instr)
        if presetdef.includes:
            for include in presetdef.includes:
                self.includeFile(include)
        if presetdef.init:
            self.csoundRenderer.addGlobalCode(presetdef.init)
        #if globalCode:
        #    self.csoundRenderer.addGlobalCode(globalCode)
        self.registeredPresets[presetdef.name] = presetdef

    def registerInstr(self, name: str, instrdef: csoundengine.Instr) -> None:
        """
        Register a csoundengine.Instr to be used with this OfflineRenderer

        .. note::

            All :class:`csoundengine.Instr` defined in the play Session are
            available to be rendered offline without the need to be registered

        Args:
            name: the name of this preset
            instrdef: the csoundengine.Instr instance

        """
        self.instrDefs[name] = instrdef
        self.csoundRenderer.registerInstr(instrdef)

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
        return self.schedEvents(events)

    def prepareSessionEvent(self, sessionevent: csoundengine.session.SessionEvent) -> None:
        """
        Prepare a session event

        Args:
            sessionevent: the session event to prepare. This is mostly used internally

        """
        pass

    def schedSessionEvent(self, event: csoundengine.session.SessionEvent
                          ) -> csoundengine.offline.ScoreEvent:
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
                          tabargs=event.tabargs,
                          **kws)

    def schedEvent(self, event: SynthEvent
                   ) -> csoundengine.offline.ScoreEvent:
        """
        Schedule a SynthEvent or a SessionEvent

        Args:
            event: a :class:`~maelzel.core.synthevent.SynthEvent`

        Returns:
            a ScoreEvent

        """
        if event.initfunc:
            event.initfunc(event, self)
        presetname = event.instr
        instrdef = self.instrDefs.get(presetname)
        if instrdef is None:
            preset = presetManager.getPreset(presetname)
            if not preset:
                raise ValueError(f"Unknown preset instr: {presetname}")
            self.preparePreset(preset, event.priority)
            instrdef = preset.getInstr()
        args = event.resolvePfields(instrdef)
        return self.csoundRenderer.sched(instrdef.name, delay=event.delay, dur=event.dur,
                                         args=args[3:], priority=event.priority,
                                         tabargs=event.args)

    def schedEvents(self,
                    coreevents: list[SynthEvent],
                    sessionevents: list[csoundengine.session.SessionEvent] = None,
                    whenfinished: Callable = None
                    ) -> list[csoundengine.offline.ScoreEvent] :
        """
        Schedule multiple events as returned by :meth:`MObj.events() <maelzel.core.MObj.events>`

        Args:
            coreevents: the maelzel.core events to schedule
            sessionevents: csound events as packed within a csoundengine.session.SessionEvent
            whenfinished: dummy arg, here to conform to the signature of the parent. Only makes
                sense in realtime

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
               for ev in coreevents]
        if sessionevents:
            out.extend(self.schedSessionEvent(ev) for ev in sessionevents)
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
        instrs.update(self.csoundRenderer.registeredInstrs())
        instrs.update(playSession().registeredInstrs())
        return instrs

    def sched(self,
              instrname: str,
              delay=0.,
              dur=-1.,
              priority=1,
              args: list[float] | dict[str, float] = None,
              whenfinished=None,
              relative=True,
              **kws) -> csoundengine.offline.ScoreEvent:
        """
        Schedule a csound event

        This method cab be used to schedule non-preset based instruments
        when rendering offline (things like global effects, for example),
        similarly to how a user might schedule a non-preset based instrument
        in real-time.

        If an OfflineRenderer is used as a context manager it is also possible
        to call the session's ._sched method directly since its _sched callback is
        rerouted to call this OfflineRenderer instead

        Args:
            instrname: the instr. name
            delay: start time
            dur: totalDuration
            priority: priority of the event
            args: any pfields passed to the instr., starting at p5
            tabargs: table args accepted by the instr.
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
        if not self.csoundRenderer.isInstrDefined(instrname):
            session = playSession()
            instr = session.getInstr(instrname)
            if not instr:
                logger.error(f"Unknown instrument {instrname}. "
                             f"Defined instruments: {self.csoundRenderer.registeredInstrs().keys()}")
                raise ValueError(f"Instrument {instrname} unknown")
            self.csoundRenderer.registerInstr(instr)
        return self.csoundRenderer.sched(instrname=instrname, delay=delay, dur=dur,
                                         priority=priority, args=args,
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
            >>> renderer._sched('reverb', priority=2)
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
        outfile, proc = self.csoundRenderer.render(outfile=outfile, wait=wait, quiet=quiet,
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
            >>> r._sched(...)
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
            outfile = _dialogs.selectFileForSave("saveCsdLastDir", filter="Csd (*.csd)")
            if not outfile:
                raise CancelledError("Save operation cancelled")
        self.csoundRenderer.writeCsd(outfile)
        return outfile

    def __enter__(self):
        """
        When used as a context manager, every call to .play will be diverted to be
        recorded offline

        """
        self._workspace = Workspace.active
        self._oldRenderer = self._workspace.renderer
        self._workspace.renderer = self

        self._session = session = playSession()
        self._oldSessionSchedCallback = session.setSchedCallback(self.sched)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            # There was an exception since entering
            logger.warning("Offline rendering aborted")
            return
        self._workspace.renderer = self._oldRenderer
        self._workspace = None
        self._oldRenderer = None

        self._session.setSchedCallback(self._oldSessionSchedCallback)
        self._oldSessionSchedCallback = None
        self._session = None

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
        return self.csoundRenderer.makeTable(data=data, size=size, sr=sr, tabnum=tabnum)

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
           extratime: float = None,
           render=True,
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
            as a context manager (see example)
        sr: sample rate of the soundfile (:ref:`config 'rec.sr' <config_rec_sr>`)
        ksmps: number of samples per cycle (:ref:`config 'rec.ksmps' <config_rec_ksmps>`)
        wait: if True, wait until recording is finished. If None,
            use the :ref:`config 'rec.blocking' <config_rec_blocking>`
        quiet: if True, supress debug information when calling
            the csound subprocess
        extratime: extra time added at the end of the render to allow
        render: if True, perform the render itself

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
        ...     r._sched('reverb', priority=2)


    See Also
    ~~~~~~~~

    :class:`OfflineRenderer`
    """
    if not events:
        # called as a context manager
        return OfflineRenderer(outfile=outfile, sr=sr, numchannels=nchnls, quiet=quiet, ksmps=ksmps)

    coreEvents, sessionEvents = _collectEvents(events, eventparams=kws, workspace=workspace)
    if not nchnls:
        nchnls = max(int(ceil(ev.resolvedPosition() + ev.chan)) for ev in coreEvents)
    renderer = OfflineRenderer(sr=sr, ksmps=ksmps, numchannels=nchnls)
    if coreEvents:
        renderer.schedEvents(coreEvents)

    if sessionEvents:
        for sessionevent in sessionEvents:
            renderer.schedSessionEvent(sessionevent)

    if extratime is None:
        cfg = getConfig()
        extratime = cfg['rec.extratime']

    if extratime:
        _, endtime = renderer.timeRange()
        endtime += extratime
    else:
        endtime = 0.
    if render:
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
        duration: the totalDuration of the test
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


def playSession() -> csoundengine.Session | None:
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
         **eventparams
         ) -> csoundengine.synth.SynthGroup | Synched:
    """
    Play a sequence of objects / events

    When playing multiple objects via their respective .play method, initialization
    (loading soundfiles, soundfonts, etc.) might result in events getting out of sync
    with each other.

    This function first collects all events; any initialization is done beforehand
    as to ensure that events keep in sync. After initialization all events are scheduled
    and their synths are gathered in a SynthGroup

    This function can also be used as context manager if not given any sources (see
    example below)

    .. note::

        To customize playback use the ``.events`` method, which works exactly like
        ``.play`` but returns the data so that it can be played later.

    Args:
        sources: a possibly nested sequence of MObjs or events as returned from
            :meth:`MObj.events`. Empty when used as a context manager.
        whenfinished: a callback taking no arguments and returning None. It will be called
            when the last event is finished
        eventparams: any keyword arguments will be passed to :meth:`MObj.events` if
            events need to be generated

    Returns:
        A SynthGroup holding all scheduled synths

    Example
    ~~~~~~~

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

    As context manager

        >>> note = Note(...)
        >>> clip = Clip(...)
        >>> with play() as p:
        ...     note.play(...)
        ...     clip.play(...)

    .. seealso::

        :class:`Synched`, :func:`render`, :meth:`MObj.play() <maelzel.core.mobj.MObj.play>`,
        :meth:`MObj.events() <maelzel.core.mobj.MObj.events>`

    """
    if not sources:
        return Synched(whenfinished=whenfinished)

    coreevents, sessionevents = _collectEvents(sources, eventparams=eventparams,
                                               workspace=Workspace.active)
    numChannels = _nchnlsForEvents(coreevents)
    if not isEngineActive():
        playEngine(numchannels=numChannels)
    else:
        engine = playEngine()
        if engine.nchnls < numChannels:
            logger.error("Some events output to channels outside of the engine's range")

    rtrenderer = RealtimeRenderer()
    return rtrenderer.schedEvents(coreevents=coreevents, sessionevents=sessionevents, whenfinished=whenfinished)


def _nchnlsForEvents(events: list[SynthEvent]) -> int:
    return max(int(ceil(ev.resolvedPosition() + ev.chan)) for ev in events)


def _schedFlatEvents(renderer: Renderer,
                     coreevents: list[SynthEvent],
                     sessionevents: list[csoundengine.session.SessionEvent] = None,
                     whenfinished: Callable = None,
                     ) -> list[csoundengine.synth.Synth]:

    for coreevent in coreevents:
        renderer.prepareEvent(coreevent)

    if sessionevents:
        for sessionevent in sessionevents:
            instr = renderer.getInstr(sessionevent.instrname)
            if instr is None:
                raise ValueError(f"Instrument {sessionevent.instrname} is not defined")
            renderer.prepareInstr(instr, priority=sessionevent.priority)

    synths = []

    resolvedArgs = [ev.resolvePfields(presetManager.presetnameToInstr(ev.instr))
                    for ev in coreevents]

    if whenfinished and renderer.isRealtime():
        lastevent = max(coreevents, key=lambda ev: ev.end if ev.end > 0 else float('inf'))
        lastevent.whenfinished = lambda id: whenfinished() if not lastevent.whenfinished else lambda id, ev=lastevent: ev.whenfinished(id) or whenfinished()

    # We take a reference time before starting scheduling,
    # so we can guarantee that events which are supposed to be
    # in sync, are in fact in sync.
    renderer.pushLock()
    for coreevent, args in zip(coreevents, resolvedArgs):
        instr = presetManager.presetnameToInstr(coreevent.instr)
        synth = renderer.sched(instr.name,
                               delay=args[0],
                               dur=args[1],
                               args=args[3:],
                               priority=coreevent.priority,
                               whenfinished=coreevent.whenfinished)
        synths.append(synth)

    if sessionevents:
        sessionsynths = [renderer.schedSessionEvent(ev)
                         for ev in sessionevents]
        synths.extend(sessionsynths)

    renderer.popLock()
    return synths


def _resolvePfields(event: SynthEvent, instr: csoundengine.Instr
                    ) -> list[float]:
    """
    returns pfields, **beginning with p2**.

    ==== =====  ======
    idx  parg    desc
    ==== =====  ======
    0    2       delay
    1    3       totalDuration
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

# TODO: rename this class and make it internal. Use `with play() as p:` instead


class Synched:
    """
    Context manager to group realtime events to ensure synched playback

    When playing multiple objects via their respective .play method, initialization
    (loading soundfiles, soundfonts, etc) might result in events getting out of sync
    with each other.

    Within this context all ``.play`` calls are collected and all events are
    scheduled at the end of the context. Any initialization is done beforehand
    as to ensure that events keep in sync. Pure csound events can also be
    scheduled in sync during this context, using the ``_sched`` method
    of the context manager or simply calling the ._sched method of
    the active session.

    After exiting the context all scheduled synths can be
    accessed via the ``synthgroup`` attribute.

    .. note::

        Use this context manager whenever you are mixing multiple objects with
        customized play arguments, and external csoundengine instruments. As an alternative
        it is possible to use :func:`play` and wrap any pure csound event into a
        :class:`csoundengine.session.SessionEvent` (see https://csoundengine.readthedocs.io/en/latest/session.html#sessionevent-class)

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
        >>> with Synched() as ctx:
        ...     chain.play(instr='piano', gain=0.5)
        ...     session._sched('reverb', 0, dur=10, priority=2)
        # Within a jupyter session placing the context after exit will display the
        # html generated by the SynthGroup generated during playback. This allows
        # outputs information about the synths and creates a Stop button to cancel
        # playback at any moment
        >>> ctx
    """
    def __init__(self, whenfinished: Callable = None):
        self.events: list[SynthEvent] = []
        """A list of all events rendered"""

        self.instrDefs: dict[str, csoundengine.Instr] = {}
        """An index of registered Instrs"""

        self.synthgroup: csoundengine.synth.SynthGroup | None = None
        """A SynthGroup holding all scheduled synths during the context"""

        self.engine: csoundengine.Engine | None = None
        """The play engine, can be used during the context"""

        self.session: csoundengine.Session | None = None
        """The corresponding Session, can be used to access the session during the context"""

        self.workspace: Workspace | None = None
        self.sessionEvents: list[csoundengine.session.SessionEvent] = []
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
        self.workspace = workspace = Workspace.active
        self._oldRenderer = workspace.renderer
        workspace.renderer = self
        self._oldSessionSchedCallback = self.session.setSchedCallback(self._sched)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Executes the operations at context end

        This includes preparing all resources and then actually
        scheduling all events
        """
        # first, restore the state prior to the context
        self.session.setSchedCallback(self._oldSessionSchedCallback)
        self._oldSessionSchedCallback = None
        self.workspace.renderer = self._oldRenderer
        self._oldRenderer = None

        if exc_type is not None:
            # There was an exception since entering
            logger.warning("Playing aborted")
            return

        if not self.events and not self.sessionEvents:
            logger.debug("No events scheduled, exiting context")
            self.synthgroup = None
            return

        r = RealtimeRenderer()
        self.synthgroup = r.schedEvents(self.events, sessionevents=self.sessionEvents,
                                        whenfinished=self._finishedCallback)

    def _sched(self,
               instrname: str,
               delay=0.,
               dur=-1.,
               priority=1,
               args: list[float] | dict[str, float] = None,
               tabargs: dict[str, float] = None,
               **kws) -> csoundengine.session.SessionEvent:
        """
        Schedule a csound event in the active Session

        This is an internal method. The user can simply call .sched on the session
        itself and it will be redirected here via its schedule callbach.

        Args:
            instrname: the instr. name
            delay: start time
            dur: totalDuration
            priority: priority of the event
            args: any pfields passed to the instr., starting at p5
            tabargs: table args accepted by the instr.
            whenfinished: dummy arg, just included to keep the signature of Session._sched
            relative: the same as whenfinished: just a placeholder
            **kws: named pfields

        Returns:
            a csoundengine's SessionEvent

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
        >>>     session.sched('reverb', 0, dur=10, priority=2, args={'kfeedback':0.9})
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
        self.sessionEvents.append(event)
        return event


