from __future__ import annotations
import csoundengine


ENGINENAME = 'maelzel.snd'


def getEngine() -> csoundengine.Engine:
    engine = csoundengine.getEngine(ENGINENAME)
    if not engine:
        engine = csoundengine.Engine(name=ENGINENAME)
    return engine
