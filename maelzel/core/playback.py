"""
This module handles playing of events

"""
from __future__ import annotations

from functools import cache
import csoundengine

from maelzel.core.workspace import Workspace
from maelzel.core import presetmanager
from maelzel.core import _playbacktools
from maelzel.core.synthevent import SynthEvent
import maelzel.core.realtimerenderer as _rtrenderer
from maelzel.core import synchronizedcontext
from maelzel.core._common import logger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import csoundengine.engine
    import csoundengine.session
    import csoundengine.baseschedevent
    import csoundengine.schedevent
    import csoundengine.tableproxy
    import csoundengine.instr
    import csoundengine.busproxy
    import csoundengine.csoundlib
    import csoundengine.synth
    import csoundengine.event
    from typing import Sequence, Callable
    from .mobj import MObj


__all__ = (
    'play',
    'testAudio',
    'audioSession',
    'audioEngine'
)


def testAudio(duration=4, period=0.5, numChannels: int | None = None, delay=0.5,
              backend=''
              ) -> None:
    """
    Test the audio engine

    Args:
        duration: the duration of the test
        period: how long to test each channel. Channels are cycled
        numChannels: the number of channels to use if starting the engine
        delay: how long to wait before starting the test.

    """
    engine = audioEngine(numchannels=numChannels, backend=backend)
    if not engine:
        engine = audioEngine(numchannels=numChannels, backend=backend)
        logger.info("Started engine, backend=%sw", engine.backend)
    engine.testAudio(dur=duration, period=period, delay=delay)


def audioDevices(backend=''
                 ) -> tuple[list[csoundengine.csoundlib.AudioDevice],
                               list[csoundengine.csoundlib.AudioDevice]]:
    """
    Returns (indevices, outdevices), where each of these lists is an AudioDevice.

    Args:
        backend: specify a backend supported by your installation of csound.
            None to use a default for you OS. Use '?' to interactively select
            a backend from a list of available options

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
    import csoundengine.csoundlib
    if backend == '?':
        backend = _selectBackend()
    return csoundengine.csoundlib.getAudioDevices(backend=backend)


def audioEngine(numchannels: int | None = None,
                backend='',
                outdev='',
                buffersize=0,
                name=''
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
        backend: the audio backend used, overrides config 'play.backend'. Use '?' to
            interactively select a backend from a list of available options
        outdev: leave as None to use the backend's default, use '?' to select
            from a list of available devices. To list all available devices
            see :func:`getAudioDevices`
        buffersize: if given, use this as the buffer size. None to use a sensible
            default for the backend

    Returns:
        the play Engine

    .. seealso:: :func:`getAudioDevices`
    """
    config = Workspace.active.config
    engineName = name or config['play.engineName']
    if engine := csoundengine.Engine.activeEngines.get(engineName):
        return engine
    if backend == "?":
        backend = _selectBackend()
    nchnls = numchannels or config['play.numChannels']
    logger.debug("Starting engine '%s' (nchnls=%d)", engineName, nchnls)
    engine = csoundengine.Engine(name=engineName,
                                 nchnls=nchnls,
                                 backend=backend or config['play.backend'],
                                 outdev=outdev,
                                 globalcode=presetmanager.presetManager.csoundPrelude,
                                 verbose=config['play.verbose'],
                                 latency=config['play.schedLatency'],
                                 buffersize=buffersize,
                                 a4=config['A4'])
    # We create the session as soon as possible, to configure the engine for
    # the session's reserved instrument ranges / tables
    session = engine.session()
    for instr in _builtinInstrs():
        session.registerInstr(instr)
    return engine


def _selectBackend() -> str:
    """
    Select a backend to use

    Returns:
        the name of the backend, or an empty string if no selection was made
    """
    import csoundengine.csoundlib
    backends = list(set(b.name for b in csoundengine.csoundlib.audioBackends()))
    backends.sort()
    from maelzel.core import _dialogs
    selectedbackend = _dialogs.selectFromList(backends, title="Select Backend")
    return selectedbackend or ''


def stopSynths():
    """
    Stops all synths (notes, chords, etc.) being played

    If stopengine is True, the play engine itself is stopped
    """
    audioSession().unschedAll(future=True)


@cache
def _builtinInstrs() -> list[csoundengine.instr.Instr]:
    from csoundengine.instr import Instr
    from csoundengine.interact import ParamSpec
    return [
        Instr('.globalreverb', r'''\
            |kfeedback=0.85, kwet=0.8, kcutoff=12000|
            if kwet > 0 then
                a1, a2 monitor
                aL, aR  reverbsc a1, a2, kfeedback, kcutoff, sr, 0.5, 1
                aL = aL * kwet - a1 * (1 - kwet) 
                aR = aR * kwet - a2 * (1 - kwet)
                outch 1, aL - a1, 2, aR - a2
            endif
            ''',
              doc="Monitor stereo reverb, affects everything"),
        Instr('.zitarev', r'''\
            |kchan=1, kwet=1, kgaindb=-12, kdelayms=60, khfdamp=6000, kdecay=3, kdamp=0.2|
            Schan1 = ".zitarev.1"
            Schan2 = ".zitarev.2"
            a1 chnget Schan1
            a2 chnget Schan2
            kinactive = detectsilence:k(a1, 0.0001, kdecay*2)
            if kinactive == 1 kgoto exit
            kdrywet = 1 - kwet
            kdecaymid = kdecay * (1 - kdamp)
            arev1, arev2 zitarev a1, a2, "drywet", kdrywet, "level", kgaindb, "delayms", kdelayms, "hfdamp", khfdamp, "decaylow", kdecay, "decaymid", kdecaymid
            outch kchan, arev1, kchan+1, arev2
            chnclear Schan1, Schan2
            exit:
        ''',
              doc="Side channel stereo reverb, applies to channels .zitarev.1 and .zitarev.2",
              properties={'kind': 'mainreverb'},
              specs=[ParamSpec('kgaindb', minvalue=-90, maxvalue=18, startvalue=-6, valuescale='log'),
                     ParamSpec('kdelayms', minvalue=0, maxvalue=400, startvalue=60),
                     ParamSpec('kdecay', minvalue=0.01, maxvalue=60, startvalue=3),
                     ParamSpec('khfdamp', minvalue=50, maxvalue=22000, startvalue=6000, valuescale='log'),
                     ParamSpec('kdamp', minvalue=0.001, maxvalue=0.999, startvalue=0.2),
                     ParamSpec('kchan', minvalue=1, maxvalue=64, startvalue=1),
                     ParamSpec('kwet', minvalue=0, maxvalue=1, startvalue=1)]
              ),
    ]


def audioSession(numchannels: int | None = None,
                 backend='',
                 outdev='',
                 buffersize: int = 0,
                 name=''
                 ) -> csoundengine.session.Session:
    """
    Returns / creates the audio Session 

    If no Session has been created already, a Session is initialized
    with the given parameters and returned. Otherwise the active
    Session is returned and any parameters passed are ignored

    .. note::
        There is one audio session, shared by all workspaces. Only the
        first call to this function will initialize the session to
        specific parameters. If you need to initialize the session
        to specific values, call this function before any playback
        related functionality is used. If any playback related
        function/method is called before, the session is created
        from default values. To configure these default values see
        the configuration

    Args:
        numchannels: the number of output channels, overrides config 'play.numChannels'
        backend: the audio backend used, overrides config 'play.backend'
        outdev: leave as None to use the backend's default, use '?' to select
            from a list of available devices. To list all available devices
            see :func:`getAudioDevices`
        buffersize: if given, use this as the buffer size. None to use a sensible
            default for the backend

    Returns:
        the active Session

    Raises:
        SessionParametersMismatchError: if ensure was True and the given parameters
            do not match the existing session

    .. seealso:: :class:`csoundengine.Session <https://csoundengine.readthedocs.io/en/latest/api/csoundengine.session.Session.html>`
    """
    if isSessionActive(name):
        return audioEngine(name=name).session()

    return audioEngine(name=name,
                       numchannels=numchannels,
                       backend=backend,
                       outdev=outdev,
                       buffersize=buffersize).session()


def isSessionActive(name='') -> bool:
    """
    Returns True if the audio session is active
    """
    if not name:
        name = Workspace.active.config['play.engineName']
    return name in csoundengine.Engine.activeEngines


def play(*sources: MObj | Sequence[SynthEvent] | csoundengine.event.Event,
         whenfinished: Callable | None = None,
         display=False,
         **eventparams
         ) -> csoundengine.synth.SynthGroup | synchronizedcontext.SynchronizedContext:
    """
    Play a sequence of objects / events in sync.  Can be used as a context manager

    When playing multiple objects via their respective .play method, initialization
    (loading soundfiles, soundfonts, etc.) might result in events getting out of sync
    with each other. This function first collects all events; any initialization is 
    done beforehand as to ensure that events keep in sync. After initialization all 
    events are scheduled and their synths are gathered in a SynthGroup

    To customize playback, use this function as a context manager or call ``.synthEvents`` 
    method on each object instead of ``.play``. ``.synthEvents`` has the same signature
    but returns the data so that it can be played later.

    Args:
        sources: a possibly nested sequence of MObjs or events as returned from
            :meth:`MObj.events`. Empty when used as a context manager.
        whenfinished: a callback taking no arguments and returning None. It will be called
            when the last event is finished
        display: if called as a context manager, the result of playback is displayed
        eventparams: any keyword arguments will be passed to :meth:`MObj.events` if
            events need to be generated

    Returns:
        A SynthGroup holding all scheduled synths

    Example
    ~~~~~~~

        >>> from maelzel.core import *
        >>> from csoundengine.session import SessionEvent
        >>> import csoundengine as ce
        >>> session = getSession()
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
        >>>     Chord("4C 4E", 7, start=1).synthEvents(position=0.5),
        >>>     Note("4C#", 6, offset=1.5),  # No customization,
        >>>     SessionEvent('reverb', dur=10, args={'kfeedback': 0.8}, priority=2),
        >>>     SessionEvent('sin', delay=0.1, dur=3, args={'imidi': 61.33, 'iamp':0.02})
        >>> )

    As context manager:

        >>> note = Note("4C#", 6, offset=1.5)
        >>> chord = Chord("4C 4E", 7, start=1)
        >>> clip = Clip(...)
        >>> with play() as s:  # returns the audio Session used
        ...     note.play(instr='.piano')
        ...     chord.play(position=0.5)
        ...     clip.play(speed=0.5, delay=1)
        ...     s.sched('reverb, priority=2')
        ...     s.sched('sin', ...)

    .. seealso::

        :func:`render`, :meth:`MObj.play() <maelzel.core.mobj.MObj.play>`,
        :meth:`MObj.synthEvents() <maelzel.core.mobj.MObj.synthEvents>`

    """
    if not sources:
        # Used as context manager
        return synchronizedcontext.SynchronizedContext(session=audioSession(),
                                                       whenfinished=whenfinished,
                                                       display=display)

    w = Workspace.active

    coreevs, sessionevs = _playbacktools.collectEvents(events=sources,
                                                       eventparams=eventparams,
                                                       workspace=w)
    numChannels = _playbacktools.nchnlsForEvents(coreevs)
    if not isSessionActive():
        engine = audioEngine(numchannels=numChannels)
    else:
        engine = audioEngine()
        if engine.nchnls < numChannels:
            logger.error("Some events output to channels outside of the engine's range")

    rtrenderer = _rtrenderer.RealtimeRenderer(engine=engine,
                                              presetManager=w.presetManager)
    return rtrenderer.schedEvents(coreevents=coreevs, sessionevents=sessionevs,
                                  whenfinished=whenfinished)



