from __future__ import annotations
import csoundengine


ENGINENAME = 'maelzel.snd'


def getEngine() -> csoundengine.Engine:
    return csoundengine.Engine.activeEngines.get(ENGINENAME) or csoundengine.Engine(name=ENGINENAME)
