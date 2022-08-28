from __future__ import annotations
from maelzel.scorestruct import ScoreStruct
from .musicobj import Note, Chord, Voice, Rest
from .score import Score
from . import symbols
from ._common import Rat, logger
import pitchtools as pt
from emlib.iterlib import pairwise
from dataclasses import dataclass
from maelzel import scoring
from typing import Callable

import xml.etree.ElementTree as ET

__all__ = (
    'parseMusicxml'
)


class MusicxmlImportError(Exception): pass


_unitToFactor = {
    'quarter': 1.,
    'eighth': 0.5,
    '16th': 0.25,
    'half': 2,
    'whole': 4,
}


def _parseMetronome(metronome: ET.Element) -> float:
    """
    Parse <metronome> tag, return the quarter-note tempo

    Args:
        metronome: the <metronome> tag element

    Returns:
        the tempo of a quarter note
    """
    unit = _.text if (_:=metronome.find('beat-unit')) else 'quarter'
    dotted = bool(metronome.find('beat-unit-dot'))
    bpm = metronome.find('per-minute').text
    factor = _unitToFactor[unit]
    if dotted:
        factor *= 1.5
    quarterTempo = float(bpm) * factor
    return quarterTempo


def _parseRest(root: ET.Element, context: dict) -> Note:
    measureRest = root.find('rest').attrib.get('measure', False) == 'yes'
    xmldur = int(root.find('duration').text)
    divisions = context['divisions']
    rest = Rest(xmldur / divisions)
    if measureRest:
        rest.properties['measureRest'] = True
    return rest


_alterToAccidental = {
    0: '',
    1: '#',
    0.5: '+',
    -1: 'b',
    -0.5: '-'
}


def _notename(step: str, octave: int, alter: float, accidental: str = '') -> str:

    cents = round(alter*100)
    if cents >= 0:
        midi = pt.n2m(f"{octave}{step}+{cents}")
    else:
        midi = pt.n2m(f"{octave}{step}-{abs(cents)}")
    notename = pt.m2n(midi)
    pitchclass = notename[1]
    if pitchclass.upper() != step.upper():
        notename = pt.enharmonic(notename)
    return notename


def _parsePitch(node) -> tuple[int, int, float]:
    step = node.find('step').text
    oct = int(node.find('octave').text)
    alter = float(_.text) if (_ := node.find('alter')) is not None else 0
    return step, oct, alter


@dataclass
class Notation:
    kind: str
    name: str
    properties: dict | None = None


def _parseNotations(root: ET.Element) -> list[Notation]:
    out = []
    node: ET.Element
    for node in root:
        if node.tag == 'articulations':
            out.extend([Notation('articulation', subnode.tag) for subnode in node])
        elif node.tag == 'ornaments':
            out.extend([Notation('ornament', subnode.tag) for subnode in node])
        elif node.tag == 'fermata':
            out.append(Notation('fermata', node.text))
    return out


def _parseNote(root: ET.Element, context: dict) -> Note:
    accidental = ''
    durationType = ''
    dur = 0
    tied = False
    chordCont = False
    properties = {}
    isRest = False
    isGrace = False
    notations = None
    annotations = []
    for node in root:
        if node.tag == 'rest':
            isRest = True
        elif node.tag == 'chord':
            chordCont = True
        elif node.tag == 'grace':
            isGrace = True
        elif node.tag == 'pitch':
            pstep, poct, palter = _parsePitch(node)
        elif node.tag == 'duration':
            dur = int(node.text) / context['divisions']
        elif node.tag == 'accidental':
            accidental = node.text
        elif node.tag == 'type':
            durationType = node.text
        elif node.tag == 'voice':
            properties['voice'] = int(node.text)
        elif node.tag == 'tie' and node.attrib.get('type', 'start') == 'start':
            tied = True
        elif node.tag == 'notations':
            notations = _parseNotations(node)
        elif node.tag == 'lyric':
            annotations.append(scoring.Annotation(node.find('text').text, placement='below'))
    if isRest:
        rest = Rest(dur)
        if properties:
            rest.properties.update(properties)
        return rest
    pitch = _notename(step=pstep, octave=poct, alter=palter)
    properties['_chordCont'] = chordCont

    if chordCont or isGrace:
        dur = 0

    note = Note(pitch=pitch, dur=dur, tied=tied, properties=properties)
    if notations:
        for notation in notations:
            if notation.kind == 'articulation':
                articulation = scoring.definitions.normalizeArticulation(notation.name)
                if articulation:
                    note.setSymbol('articulation', notation.name)
                else:
                    # If this is an unsupported articulation, at least save it as a property
                    note.properties['mxml/articulation'] = notation.name
            elif notation.kind == 'ornament':
                ornament = scoring.definitions.normalizeOrnament(notation.name)
                if ornament:
                    note.setSymbol('ornament', ornament)
                else:
                    note.properties['mxml/ornament'] = notation.name
            elif notation.kind == 'fermata':
                note.setSymbol('fermata', scoring.definitions.normalizeFermata(notation.name))
    for annotation in annotations:
        note.setSymbol(symbols.Expression(text=annotation.text, placement=annotation.placement))

    return note


def _joinChords(notes: list[Note]) -> list[Note|Chord]:
    """
    Join notes belonging to a chord

    Musicxml encodes chords as individual notes, where
    the first note of a chord is just a regular note
    followed by other notes which contain the <chord/>
    tag. Those notes should be merged to the previous
    note into a chord. The duration is given by the
    first note and no subsequent note can be longer
    (but they might be shorted).

    Since at the time in maelzel.core all notes within
    a chord share the same duration we discard
    all durations but the first one.

    Args:
        notes: a list of Notes, as parsed from musicxml.

    Returns:
        a list of notes/chords
    """
    # mark chord starts
    groups = []
    for note, nextnote in pairwise(notes):
        if not note.properties.get('_chordCont') and nextnote.properties.get('_chordCont'):
            groups.append([note])
        elif note.properties.get('_chordCont'):
            groups[-1].append(note)
        else:
            groups.append(note)
    out = []
    for group in groups:
        if isinstance(group, Note):
            out.append(group)
        else:
            first = group[0]
            assert isinstance(first, Note)
            chord = Chord(group, dur=first.dur, dynamic=first.dynamic, tied=first.tied, start=first.start)
            chord.properties['voice'] = first.properties.get('voice', 1)
            out.append(chord)
    return out


def _measureDuration(beats: int, beattype: int) -> Rat:
    return Rat(beats*4, beattype)


@dataclass
class Direction:
    kind: str
    value: str
    placement: str = ''
    properties: dict | None = None


def _attr(attrib: dict, key: str, default, convert=None):
    value = attrib.get(key)
    if value is not None:
        return value if not convert else convert(value)
    return default


def _parsePosition(x: ET.Element) -> str:
    attrib = x.attrib
    defaulty = _attr(attrib, 'default-y', 0., float)
    relativey = _attr(attrib, 'relative-y', 0., float)
    pos = defaulty + relativey
    return '' if pos == 0 else 'above' if pos > 0 else 'below'


def _parseAttribs(attrib: dict, keys: dict[str, Callable]) -> dict:
    out = {}
    for k, convertfunc in keys.items():
        v = attrib.get(k)
        if v is not None:
            out[k] = v if not convertfunc else convertfunc(v)
    return out


def _parsePart(part: ET.Element, context: dict) -> tuple[dict[int, Voice], ScoreStruct]:
    """
    Parse a part

    Args:
        part: the <part> subtree

    Returns:
        a tuple (dict of voices, scorestruct)
    """
    sco = ScoreStruct()
    quarterTempo = 60
    beats, beattype = 4, 4
    if (divisionsTag := part.find('./measure/attributes/divisions')) is not None:
        divisions = int(divisionsTag.text)
        if divisions != context['divisions']:
            context = context.copy()
            context['divisions'] = divisions

    voices: dict[int, list[Note]] = {}
    measureCursor = Rat(0)
    for measureidx, measure in enumerate(part.findall('measure')):
        directions: list[Direction] = []
        cursors = {voicenum: measureCursor for voicenum in range(10)}

        # we find first the metronome, if present, so that we can add a new
        # measure definition if there is a change in time-signature
        # NB: we do not parse <sound tempo=".."> tags
        if metronome := measure.find('./direction/direction-type/metronome'):
            quarterTempo = _parseMetronome(metronome)

        if time := measure.find('./attributes/time'):
            beats = int(time.find('beats').text)
            beattype = int(time.find('beat-type').text)

        sco.addMeasure(timesig=(beats, beattype), quarterTempo=quarterTempo)

        for item in measure:
            tag = item.tag
            if tag == 'direction':
                inner = item.find('direction-type')[0]
                if inner.tag == 'dynamics':
                    dynamic = inner[0].tag
                    dynamic2 = scoring.definitions.normalizeDynamic(dynamic)
                    if dynamic2:
                        directions.append(Direction('dynamic', dynamic2))
                    else:
                        directions.append(Direction('words', dynamic, placement='below',
                                                    properties={'font-style': 'italic'}))
                elif inner.tag == 'words':
                    # TODO: parse style / font / etc.
                    placement = _parsePosition(inner)
                    properties = _parseAttribs(inner.attrib, {'font-size': float, 'font-style':None})
                    directions.append(Direction('words', inner.text, placement=placement,
                                                properties=properties))
                elif inner.tag == 'rehearsal':
                    enclosed = bool(inner.attrib.get('enclosed'))
                    directions.append(Direction('rehearsal', inner.text,
                                                properties={'enclosed': enclosed}))

            elif tag == 'note':
                note = _parseNote(item, context)
                voicenum = note.properties.get('voice', 1)
                cursor = cursors[voicenum]
                cursorWithinMeasure = cursor - measureCursor
                note.start = cursor
                for direction in directions:
                    if direction.kind == 'dynamic':
                        note.dynamic = direction.value
                    elif direction.kind == 'words':
                        note.addText(direction.value,
                                     placement=direction.placement,
                                     fontsize=direction.properties.get('font-size'),
                                     fontstyle=direction.properties.get('font-style'))
                    elif direction.kind == 'rehearsal':
                        if cursorWithinMeasure == 0:
                            sco.addRehearsalMark(measureidx, direction.value)
                        else:
                            # A rehearsal mark in the middle of the measure?? This
                            # can be added as a text annotation
                            note.addText(direction.value,
                                         placement='above',
                                         fontstyle='bold',
                                         box=True)
                voices.setdefault(voicenum, []).append(note)
                cursors[voicenum] += note.dur
                directions.clear()

        # ...
        measureCursor += Rat(beats*4, beattype)

    if len(sco) == 0:
        sco.addMeasure((4, 4), quarterTempo=60)
    return {voicenum: Voice(_joinChords(notes)) for voicenum, notes in voices.items()}, sco


def parseMusicxml(xml: str) -> Score:
    metadata = {}
    root = ET.fromstring(xml)
    if root.tag != 'score-partwise':
        raise MusicxmlImportError(f'Only score-partwise format is supported, got {root.tag}')
    version = root.get('version')
    if version:
        metadata['mxml/version'] = version

    # Parse metadata
    if movementTitle := root.find('movement-title'):
        metadata['mxml/movement-title'] = movementTitle.text

    if identification := root.find('identification'):
        if creator := identification.find('creator'):
            creatortype = creator.get('type')
            metadata[f'mxml/identification/{creatortype}'] = creator.text

    # part list
    partlist = root.find('part-list')
    if not partlist:
        raise MusicxmlImportError("No part-list tag found")

    partsRegistry = {}
    for item in partlist:
        if item.tag == 'score-part':
            partid = item.get('id')
            partdict = {'xml': item, 'id': partid}
            if name := item.find('part-name'):
                partdict['name'] = name
            partsRegistry[partid] = partdict

    # first, find the divisions. They must be defined in at least one part
    divisionsNode = root.find('./part/measure/attributes/divisions')
    if divisionsNode is None:
        raise MusicxmlImportError("This score does not define divisions")
    divisions = int(divisionsNode.text)
    context = {'divisions': divisions}

    scorestructs: list[ScoreStruct] = []
    allvoices: list[Voice] = []
    for partidx, part in enumerate(root.findall('part')):
        partid = part.get('id')
        logger.debug("Parsing part", partidx, partid)
        if partid is None:
            raise MusicxmlImportError(f"Part definition does not have an id: {part}")
        voicesdict, scorestruct = _parsePart(part, context)
        partdef = partsRegistry[partid]
        voicename = partdef.get('part-name', partid)
        if len(voicesdict) == 1 and 1 in voicesdict:
            # Only one voice
            voicesdict[1].label = voicename
        else:
            for voicenum, voice in voicesdict.items():
                voice.label = f'{voicename}/{voicenum}'
        allvoices.extend(voicesdict.values())
        scorestructs.append(scorestruct)

    scorestruct = max(scorestructs, key=lambda s: len(s))
    scorestruct.title = metadata.get('mxml/movement-title', '')
    scorestruct.composer = metadata.get('mxml/identification/composer', '')
    score = Score(voices=allvoices, scorestruct=scorestruct)
    score.properties.update(metadata)
    return score