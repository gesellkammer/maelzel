import csoundengine


CSOUNDENGINE = 'maelzel.snd'


def getEngine() -> csoundengine.Engine:
    """
    Create the default engine for this module, or return an already active one

    Returns:
        the created/active Engine
    """
    engine = csoundengine.getEngine(CSOUNDENGINE)
    if engine is not None:
        return engine
    return csoundengine.Engine(name=CSOUNDENGINE)