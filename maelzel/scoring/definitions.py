from __future__ import annotations

stemTypes = {
    'normal',
    'hidden'
}

availableArticulations = {'accent',         # >
                          'staccato',       # .
                          'tenuto',         # -
                          'marcato',        # ^
                          'staccatissimo',  # ' wedge
                          'espressivo',     # <>
                          'portato'         # - + .
                          }

"""
musicxml articulations
(https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/articulations/)

    <accent>
    <strong-accent>
    <staccato>
    <tenuto>
    <detached-legato>
    <staccatissimo>
    <spiccato>
    <scoop>
    <plop>
    <doit>
    <falloff>
    <breath-mark>
    <caesura>
    <stress>
    <unstress>
    <soft-accent>
    <other-articulation>
"""


articulationMappings = {
    'strong-accent': 'marcato',
    'soft-accent': 'espressivo',
    'detached-legato': 'portato',
    'spiccato': 'staccatissimo'
}


def normalizeArticulation(articulation: str, default='') -> str:
    if articulation in availableArticulations:
        return articulation
    elif (mapped := articulationMappings.get(articulation)) is not None:
        return mapped
    else:
        return default


noteheadShapes = {
    'cross',
    'harmonic',
    'triangleup',
    'xcircle',
    'triangle',
    'rhombus',
    'square',
    'rectangle'
}


# These dynamics are supported in both lilypond and musicxml
dynamicLevelsSeq = ["ppppp", "pppp", "ppp", "pp", "p", "mp", "mf",
                    "f", "ff", "fff", "ffff", "fffff"]


dynamicLevels = set(dynamicLevelsSeq)


# These expressions are supported by lilypond
dynamicExpressions = {
    'fp', 'sf', 'sff', 'sp', 'spp', 'sfz', 'rfz', 'n'
}

# Dynamics which are not supported should be converted to the nearest
# supported dynamic. This is particularly important when parsing some
# external source, like muscxml

dynamicMappings = {
    'pppppp': 'ppppp',
    'ffffff': 'fffff',
    'sfp': 'sp',
    'sfpp': 'spp',
    'rf': 'rfz',
    'sffz': 'sfz',
    'fz': 'sf',
    'pf': 'f',
    'sfzp': 'fp'
}


def normalizeDynamic(dynamic: str, default='') -> str:
    """
    Normalize a dynamic, returns *default* if the dynamic is invalid

    Args:
        dynamic: the dynamic to normalize

    Returns:
        the normalized dynamic
    """
    if dynamic in dynamicLevels or dynamic in dynamicExpressions:
        return dynamic
    elif (mapped := dynamicMappings.get(dynamic)) is not None:
        return mapped
    return default


availableOrnaments = {'trill', 'mordent', 'prall'}


ornamentMappings = {
    'trill-mark': 'trill',
    'inverted-mordent': 'prall'
}


def normalizeOrnament(ornament: str, default='') -> str:
    if ornament in availableOrnaments:
        return ornament
    elif (_:=ornamentMappings.get(ornament)) is not None:
        return _
    return default


availableFermatas = {
    'normal',
    'square',
    'angled',
    'double-angled',
    'double-square'
}


def normalizeFermata(fermata: str) -> str:
    return fermata if fermata in availableFermatas else 'normal'
