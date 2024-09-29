import csoundengine


CSOUNDENGINE = 'maelzel.snd'


def getEngine() -> csoundengine.Engine:
    """
    Create the default engine for this module, or return an already active one

    Returns:
        the created/active Engine
    """
    return csoundengine.Engine.activeEngines.get(CSOUNDENGINE) or csoundengine.Engine(name=CSOUNDENGINE)