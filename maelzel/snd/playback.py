from __future__ import annotations
import csoundengine


ENGINENAME = 'maelzel.snd'


def getEngine() -> csoundengine.Engine:
    engine = csoundengine.getEngine('maelzel.snd')
    if not engine:
        engine = csoundengine.Engine(name='maelzel.snd')
    return engine