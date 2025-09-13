"""

CoreConfig: Configuration for maelzel.core
============================w==============

At any given moment there is one active configuration (an instance of :class:`CoreConfig`,
itself a subclass of `dict`).
The configuration allows to set default values for many settings and customize different
aspects of **maelzel.core**:

* notation (default page size, rendered image scaling, etc). Prefix: *show*
* playback (default audio backend, instrument, etc). Prefix: *play*
* offline rendering. Prefix: *rec*
* quantization (complexity, quantization strategy, etc). Prefix: *quant*
* etc.

Settings can be modified by simply changing the values of the active config dict::

    # Get the active config
    >>> from maelzel.core import *
    >>> config = getConfig()
    >>> config['show.pageSize'] = 'A3'

A config has a :ref:`set of valid keys <coreconfigkeys>`. An attempt to set an unknown key will
result in an error. Values are also validated regarding their type and accepted
values, range, etc.::

    >>> config['foo'] = 'bar'
    KeyError: 'Unknown key foo'
    >>> config['show.pageSize'] = 'Z1'
    ValueError: key show.pageSize should be one of {'a2', 'a4', 'a3'}, got Z1


Alternative key format
----------------------

For convenience keys are case- and punctuation- independent. This allows
to create a new CoreConfig as ``CoreConfig(show_staff_size=10)`` instead of
``CoreConfig(updates={'show.staffSize': 10})`` or query the same key as
``staffsize = config['show_staff_size']`` instead of ``staffsize = config['show.staffSize']``


Persistence
-----------

Modifications to a configuration can be made persistent by saving the config.

    >>> from maelzel.core import *
    # Set the reference frequency to 443
    >>> conf = getConfig()
    >>> conf['A4'] = 443
    # Set lilypond as default rendering backend
    >>> conf['show.backend'] = 'lilypond'
    >>> conf.save()

In a future session these changes will be picked up as default:

    >>> from maelzel.core import *
    >>> conf = getConfig()
    >>> conf['A4']
    443

.. seealso::

    :ref:`workspace_mod`

--------------------

.. _rootconfig:

Root Config
-----------

When ``maelzel.core`` is first imported the persisted config is read (if no persisted
config is found the builtin default is used). This is the *root config* and is used as a
prototype for any subsequent :class:`CoreConfig` created. This enables you to
modify default values based on your personal setup (for example, you can set the
default rendering samplerate to 48000 if that fits your workflow better).

Example
~~~~~~~

    >>> from maelzel.core import *
    >>> config = getConfig()
    >>> config['rec.sr']
    44100
    >>> rootConfig['rec.sr'] = 48000
    >>> newconfig = CoreConfig()
    >>> newconfig['rec.sr']
    48000
    >>> rootConfig.save()   # If you want this as default, save it for any future sessions

---------------------

.. _activeconfig:

Active config
-------------

In order to create a configuration specific for a particular task it is possible
to create a new config with :class:`~maelzel.core.config.CoreConfig` or by
cloning any CoreConfig.

    >>> from maelzel.core import *
    >>> newconfig = CoreConfig({'show.pageSize': 'a3'}, active=True)

Also creating a new :class:`~maelzel.core.workspace.Workspace` will create a new
config based on the root config:

    >>> from maelzel.core import *
    # Create a config to work with old tuning and display notation using a3 page size
    >>> w = Workspace(updates={'A4': 435, 'show.pageSize': 'a3'}, active=True)
    # do something with this, then deactivate the workspace
    >>> n = Note("4A")
    >>> print(n.freq)
    435
    # Play with default instr
    >>> n.play()
    # When finished, deactivate it to return to previous Workspace
    >>> w.deactivate()
    >>> Note("4A").freq
    442

It is also possible to create a temporary config as a context manager. The config
will be active only within its context:

    >>> from maelzel.core import *
    >>> scale = Chain([Note(m, dur=0.5) for m in range(60, 72)])
    >>> with CoreConfig({'show.pageSize':'a3'}):
    ...     scale.show()
"""
from __future__ import annotations

import os

import configdict
from configdict import ConfigDict
from maelzel.core._common import logger
from maelzel.core import configdata

import typing as _t
if _t.TYPE_CHECKING:
    from maelzel.scoring.render import RenderOptions
    from maelzel.scoring.quantprofile import QuantizationProfile
    import maelzel.scoring.enharmonics as enharmonics
    from maelzel.core.workspace import Workspace


__all__ = (
    'CoreConfig',
)


class _UNKNOWN:
    pass


def _syncCsoundengineTheme(theme: str):
    import csoundengine
    csoundengine.config['html_theme'] = theme


def _resetImageCacheCallback(config: CoreConfig, force=False):
    from . import workspace
    if force or config is workspace.getConfig():
        from . import mobj
        mobj.clearImageCache()


def _propagateA4(config: CoreConfig, a4: float) -> None:
    from . import workspace
    w = workspace.Workspace.active
    assert w is not None
    # Is it the active config?
    if config is w.config:
        w.a4 = a4


#####################################
#            CoreConfig             #
#####################################


class CoreConfig(ConfigDict):
    """
    A CoreConfig is a ``dict`` like object which controls many aspects of **maelzel.core**

    A **CoreConfig** reads its settings from a persistent copy. This persistent version is
    generated whenever the user calls the method :meth:`CoreConfig.save`.
    When **maelzel.core** is imported it reads this configuration and creates
    the **root**, which is an instance of :class:`CoreConfig` and can be accessed
    via the class variable :attr:``CoreConfig.root``

    Notice that a configuration, in order to modify the behaviour of the environment,
    needs to be either actively used (passed as an argument to any function accepting
    a configuration object) or set as active via :func:`~maelzel.core.workspace.setConfig`
    or by calling its :meth:`CoreConfig.activate` method.

    Args:
        updates: if given, a dict which will be used to update the newly created instance
        source: either a ConfigDict to use as prototype; 'root', to use the root
            config (the last saved config); 'load' to reload the last saved config
            This ConfigDict will be a copy of that prototype
        active: if True, set this CoreConfig as active (modifying the current Workspace)
        kws: any keywords will be used to update the config and must be valid keys for
            a CoreConfig

    .. admonition:: See Also

        :ref:`Configuration Keys <coreconfigkeys>` for documentation on the keys and
        their possible values

    .. seealso:: :func:`maelzel.core.workspace.makeConfig`

    """
    _root: _t.ClassVar[CoreConfig | None] = None
    _defaultName: _t.ClassVar[str] = 'maelzel.core'
    _keyToType: _t.ClassVar[dict[str, type | tuple[type, ...]]] = {}
    _listHiddenKeysAtTheEnd: _t.ClassVar[bool] = True

    # A config callback has the form (config: CoreConfig, key: str, val: Any) -> None
    # It is called with the config being modified, the key being modified and the new value
    _builtinCallbacks = {
        'htmlTheme': lambda config, key, val: _syncCsoundengineTheme(val),
        r"(show|quant|\.quant)\..+": lambda config, key, val: _resetImageCacheCallback(config, force=True),
        "A4": lambda config, key, val: _propagateA4(config, val),
    }

    def __init__(self,
                 updates: dict[str, _t.Any] | None = None,
                 source: ConfigDict | str = 'root',
                 active=False,
                 **kws):
        self._hash: int = 0
        self._defaultPlayArgsDict: dict | None = None

        load = source == 'load' or (source == 'root' and CoreConfig._root is None)

        super().__init__(CoreConfig._defaultName,
                         default=configdata.defaultdict,
                         persistent=False,
                         validator=configdata.validator,
                         docs=configdata.docs,
                         load=load,
                         strict=False)

        if not load:
            if source == 'root':
                if not hasattr(CoreConfig, 'root') or CoreConfig._root is None:
                    raise RuntimeError("CoreConfig not initialized!")
                source = CoreConfig._root
            if isinstance(source, ConfigDict):
                d = dict(source)
                dict.update(self, d)
        else:
            # Whenever loading, update root
            CoreConfig._root = self

        self._previousState: tuple[Workspace, CoreConfig] | None = None

        self.readonly = False
        """If True, trying to modify this dict will raise a ReadOnlyException"""

        for regex, func in self._builtinCallbacks.items():
            self.registerCallback(func, pattern=regex)

        self.registerCallback(self._changedCallback)

        if updates:
            self.update(updates)

        if kws:
            kws = self._normalizeDict(kws)
            kws = {k: v for k, v in kws.items()
                   if k in self.keys()}
            self.update(kws)

        if active:
            self.activate()

    @classmethod
    def root(cls) -> CoreConfig:
        root = CoreConfig._root
        assert root is not None
        return root

    def _changedCallback(self, cfg: ConfigDict, key: str, val):
        self._hash = 0
        self._defaultPlayArgsDict = None

    def _ipython_key_completions_(self):
        return self.keys()

    def _repr_keys(self) -> list[str]:
        # This places "hidden" entries (entries starting with a dot) at the end
        # of any list
        if not CoreConfig._listHiddenKeysAtTheEnd:
            return super()._repr_keys()
        lastchr = chr(9999)
        return sorted(self.keys(), key=lambda k: k if not k.startswith(".") else lastchr + k)

    @classmethod
    def read(cls, path: str):
        """
        Create a new CoreConfig from the saved config

        The path points to a .yaml config saved via :meth:`CoreConfig.save`

        Args:
            path: the path to a config

        Returns:
            the new CoreConfig
        """
        out = CoreConfig()
        out.load(path)
        return out

    def copy(self) -> CoreConfig:
        """Create a copy of this config"""
        return CoreConfig(source=self)

    def clone(self, updates: dict | None = None, **kws) -> CoreConfig:
        """
        Create a new CoreConfig from this config with modified values

        Keywords which are not valid symbols (keys containing periods, like
        'quant.nestedTuples') can be used by removing any invalid characters.
        Also, config keys are not case sensitive. In the case above, ``quantNestedTuples``
        could be used

        .. note::

            A :class:`CoreConfig` can be used as a context manager to activate
            it temporarily. Used together with this method it can be used
            to modify any rendering or playback condition for the code
            within that block.

        Args:
            updates: a dictionary of updates to apply to the new config
            **kws: keyword arguments to pass to the new config.

        Returns:
            the new CoreConfig

        Example
        ~~~~~~~

        Render an object with modified quantization

        .. code-block:: python

            >>> from maelzel.core import *
            >>> cfg = getConfig()
            >>> with cfg.clone(quantNestedTuples=False):
            ...     Chain(...).show()

        """
        if kws:
            kws = self._normalizeDict(kws)

        return CoreConfig(updates=updates, source=self, **kws)

    def getType(self, key: str) -> type | tuple[type, ...]:
        if (t := self._keyToType.get(key, _UNKNOWN)) is not _UNKNOWN:
            return t
        t = super().getType(key)
        self._keyToType[key] = t
        return t

    def __hash__(self):
        if self._hash:
            return self._hash
        self._hash = super().__hash__()
        return self._hash

    def makeRenderOptions(self) -> RenderOptions:
        """
        Create RenderOptions based on this config

        Returns:
            a RenderOptions instance
        """
        from maelzel.core import notation
        return notation.makeRenderOptionsFromConfig(self)

    def makeQuantizationProfile(self) -> QuantizationProfile:
        """
        Create a QuantizationProfile from this config

        """
        from maelzel.core import notation
        return notation.makeQuantizationProfileFromConfig(self)
        
    def makeEnharmonicOptions(self) -> enharmonics.EnharmonicOptions:
        """
        Create EnharmonicOptions from this config

        The returned object is used within maelzel.scoring.enharmonics to
        determine the best
        to

        Returns:
            a :class:`maelzel.scoring.enharmonics.EnharmonicOptions`
        """
        from maelzel.core import notation
        return notation.makeEnharmonicOptionsFromConfig(self)

    def __enter__(self):
        from . import workspace
        w = workspace.Workspace.active
        self._previousState = (w, w.config)
        w.config = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._previousState is not None
        workspace, prevconfig = self._previousState
        assert workspace.isActive() and workspace.config is self and prevconfig
        workspace.config = prevconfig

    def activate(self) -> None:
        """
        Make this config the active config

        This is just a shortcut for ``setConfig(self)``
        """
        from . import workspace
        active = workspace.Workspace.active
        assert active is not None
        active.config = self

    def saveKey(self, key: str) -> None:
        conf = CoreConfig()
        conf[key] = self[key]
        conf.save()

    def reset(self, removesaved=False) -> None:
        """
        Reset this config to its defaults

        Args:
            removesaved: if True, remove any saved config

        """
        super().reset()
        from maelzel.core.presetmanager import presetManager
        if '.piano' in presetManager.presetdefs:
            self['play.instr'] = '.piano'

        if removesaved:
            path = self.getPath()
            if os.path.exists(path):
                os.remove(path)

    @classmethod
    def removeSaved(cls):
        """Remove the saved default config"""
        path = configdict.configPathFromName(cls._defaultName)
        if os.path.exists(path):
            logger.debug(f"Removing default config at '{path}'")
            os.remove(path)

    def _makeDefaultPlayArgsDict(self, copy=True) -> dict:
        """
        Creates the dict for a default PlayArgs

        This is used as the base for each event created

        Args:
            copy: if True, the returned dict is a copy and can be modified

        Returns:
            a dict to be passed to PlayArgs


        """
        if self._defaultPlayArgsDict is not None:
            d = self._defaultPlayArgsDict
        else:
            d = dict(delay=0,
                     chan=1,
                     gain=self['play.gain'],
                     fade=self['play.fade'],
                     instr=self['play.instr'],
                     pitchinterpol=self['play.pitchInterpol'],
                     fadeshape=self['play.fadeShape'],
                     priority=1,
                     position=-1,
                     sustain=0,
                     transpose=0)
            self._defaultPlayArgsDict = d
        return d.copy() if copy else d
