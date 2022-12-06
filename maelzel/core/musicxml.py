from __future__ import annotations
import copy
from maelzel.scorestruct import ScoreStruct, TimeSignature
from .event import Note, Chord, Rest
from .chain import Voice
from .score import Score
from . import symbols
from maelzel.common import F
from ._common import logger
import pitchtools as pt
from emlib.iterlib import pairwise
import emlib.mathlib
from dataclasses import dataclass
from maelzel import scoring
from typing import Callable

import xml.etree.ElementTree as ET


__all__ = (
    'parseMusicxml',
    'parseMusicxmlFile',
    'MusicxmlParseError',
)


class MusicxmlParseError(Exception): pass


_unitToFactor = {
    'quarter': 1.,
    'eighth': 0.5,
    '16th': 0.25,
    'half': 2,
    'whole': 4,
}


class ParseContext:
    def __init__(self, divisions: int, fixSpelling=True):
        self.divisions = divisions
        self.fixSpelling = fixSpelling
        self.spanners: dict[str, symbols.Spanner] = {}
        self.octaveshift: int = 0
        self.transposition: int = 0

    def copy(self) -> ParseContext:
        out = copy.copy(self)
        assert out.divisions > 0
        return out


def _parseMetronome(metronome: ET.Element) -> float:
    """
    Parse <metronome> tag, return the quarter-note tempo

    Args:
        metronome: the <metronome> tag element

    Returns:
        the tempo of a quarter note
    """
    unit = _.text if (_:=metronome.find('beat-unit')) is not None else 'quarter'
    dotted = bool(metronome.find('beat-unit-dot') is not None)
    bpm = metronome.find('per-minute').text
    factor = _unitToFactor[unit]
    if dotted:
        factor *= 1.5
    quarterTempo = float(bpm) * factor
    return quarterTempo


def _parseRest(root: ET.Element, context: ParseContext) -> Note:
    measureRest = root.find('rest').attrib.get('measure', False) == 'yes'
    xmldur = int(root.find('duration').text)
    divisions = context.divisions
    rest = Rest(dur=F(xmldur, divisions))
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


def _parsePitch(node, prefix='') -> tuple[int, int, float]:
    step = node.find(f'{prefix}step').text
    oct = int(node.find(f'{prefix}octave').text)
    alter = float(_.text) if (_ := node.find('alter')) is not None else 0
    return step, oct, alter


def _makeSpannerId(prefix: str, d: dict, key='number'):
    spannerid = d.get(key)
    return prefix if not spannerid else f"{prefix}-{spannerid}"


@dataclass
class Notation:
    kind: str
    value: str = ''
    properties: dict | None = None
    skip: bool = False


def _makeSpannerNotation(kind: str, class_: type, node: ET.Element):
    properties = {k: v for k, v in node.attrib.items()}
    properties['class'] = class_
    properties['mxml/tag'] = f'notation/{node.tag}'
    return Notation('spanner', kind, properties=properties)


def _parseOrnament(node: ET.Element) -> Notation:
    tag = node.tag
    if tag == 'tremolo':
        # <tremolo type="start">2</tremolo>
        # <tremolo default-x="6" default-y="-27" type="single">3</tremolo>
        tremtype = node.attrib.get('type')
        if tremtype == 'stop':
            tremtype = 'end'
        return Notation('ornament', 'tremolo',
                        properties={'tremtype': tremtype,
                                    'nummarks': int(node.text)})
    else:
        return Notation('ornament', tag, properties=dict(node.attrib))


def _parseNotations(root: ET.Element) -> list[Notation]:
    notations = []
    node: ET.Element
    for node in root:
        tag = node.tag
        assert isinstance(node.attrib, dict)
        if tag == 'articulations':
            notations.extend([Notation('articulation', subnode.tag, properties=dict(subnode.attrib)) for subnode in node])
        elif tag == 'ornaments':
            notations.extend([_parseOrnament(subnode) for subnode in node])
        elif tag == 'glissando':
            notation = _makeSpannerNotation('glissando', symbols.NoteheadLine, node)
            notations.append(notation)
        elif tag == 'slide':
            notation = _makeSpannerNotation('glissando', symbols.NoteheadLine, node)
            notations.append(notation)
        elif tag == 'fermata':
            notations.append(Notation('fermata', node.text, properties=dict(node.attrib)))
        elif tag == 'dynamics':
            dyn = node[0].tag
            if dyn == 'other-dynamics':
                dyn = node[0].text
            notations.append(Notation('dynamic', dyn))
        elif tag == 'arpeggiate':
            notations.append(Notation('articulation', 'arpeggio'))
        elif tag == 'technical':
            notation = _parseTechnicalNotation(node)
            if notation is not None:
                notations.append(notation)
        elif tag == 'slur':
            notation = _makeSpannerNotation('slur', symbols.Slur, node)
            notations.append(notation)
    out = []

    # Handle some special cases
    if notations:
        for n0, n1 in pairwise(notations):
            if (n0.kind == 'ornament' and
                    n1.kind == 'ornament' and
                    n0.value == 'trill-mark' and
                    n1.value == 'wavy-line'):
                n1.properties['startmark'] = 'trill'
            elif (n0.kind == 'ornament' and n1.kind == 'ornament' and
                n0.value == 'wavy-line' and n1.value == 'wavy-line' and
                n0.properties['type'] == 'start' and n1.properties['type'] == 'stop'):
                n1.value = 'inverted-mordent'
            else:
                out.append(n0)
        out.append(notations[-1])
    return out


_technicalNotationToArticulation = {
    'up-bow': 'upbow',
    'down-bow': 'downbow',
    'open-string': 'openstring'
}


def _parseTechnicalNotation(root: ET.Element) -> Notation | None:
    inner = root[0]
    tag = inner.tag
    if tag == 'stopped':
        return Notation('articulation', 'closed')
    elif tag == 'snap-pizzicato':
        return Notation('articulation', 'snappizz')
    elif tag == 'string':
        whichstr = inner.text
        if whichstr:
            romannum = emlib.mathlib.roman(int(whichstr))
            return Notation('text', romannum, properties={'placement': 'above'})
    elif tag == 'harmonic':
        # possibilities:
        # harmonic
        # harmonic/natural
        # harmonic/artificial
        if len(inner) == 0:
            return Notation('articulation', 'flageolet')
        elif inner[0].tag == 'artificial':
            return Notation('notehead', 'harmonic')
        elif inner[0].tag == 'natural':
            if inner.find('touching-pitch') is not None:
                # This should be solved via a notehead change so leave this out
                return Notation('notehead', 'harmonic')
            elif inner.find('base-pitch') is not None:
                # Do nothing here
                pass
            else:
                return Notation('articulation', 'flageolet')
    elif (articulation := _technicalNotationToArticulation.get(tag)) is not None:
        return Notation('articulation', value=articulation)
    elif tag == 'fingering':
        return Notation('fingering', value=inner.text)
    elif tag == 'bend':
        bendalter = float(inner.find('bend-alter').text)
        return Notation('bend', properties={'alter': bendalter})


def _parseNote(root: ET.Element, context: ParseContext) -> Note:
    notesymbols = []
    pstep = ''
    dur = 0
    noteType = ''
    tied = False
    properties = {}
    notations = []
    for node in root:
        if node.tag == 'rest':
            noteType = 'rest'
        elif node.tag == 'chord':
            noteType = 'chord'
        elif node.tag == 'unpitched':
            noteType = 'unpitched'
            pstep, poct, palter = _parsePitch(node, prefix='display-')
        elif node.tag == 'notehead':
            shape = scoring.definitions.normalizeNoteheadShape(node.text)
            parens = node.attrib.get('parentheses') == 'yes'
            if not shape:
                logger.warning(f'Notehead shape not supported: "{node.text}"')
            else:
                notesymbols.append(symbols.Notehead(shape=shape, parenthesis=parens))
            properties['mxml/notehead'] = node.text
        elif node.tag == 'grace':
            noteType = 'grace'
        elif node.tag == 'pitch':
            pstep, poct, palter = _parsePitch(node)
        elif node.tag == 'duration':
            dur = F(int(node.text), context.divisions)
        elif node.tag == 'accidental':
            accidental = node.text
            properties['mxml/accidental'] = accidental
            if node.attrib.get('editorial') == 'yes':
                notesymbols.append(symbols.Accidental(parenthesis=True))
        elif node.tag == 'type':
            properties['mxml/durationtype'] = node.text
        elif node.tag == 'voice':
            properties['voice'] = int(node.text)
        elif node.tag == 'tie' and node.attrib.get('type', 'start') == 'start':
            tied = True
        elif node.tag == 'notations':
            notations.extend(_parseNotations(node))
        elif node.tag == 'lyric':
            if (textnode:=node.find('text')) is not None:
                text = textnode.text
                if text:
                    notesymbols.append(symbols.Text(text, placement='below'))
            else:
                ET.dump(node)
                logger.error("Could not find lyrincs text")


    if noteType == 'rest':
        rest = Rest(dur)
        if properties:
            if rest.properties is None:
                rest.properties = properties
            else:
                rest.properties.update(properties)
        return rest

    if not pstep:
        ET.dump(root)
        raise MusicxmlParseError("Did not find pitch-step for note")

    # notename = _notename(step=pstep, octave=poct + context.octaveshift, alter=palter)
    notename = _notename(step=pstep, octave=poct, alter=palter)
    if context.transposition != 0:
        notename = pt.transpose(notename, context.transposition)

    if noteType == 'chord':
        dur = 0
        properties['_chordCont'] = True
    elif noteType == 'grace':
        dur = 0

    note = Note(pitch=notename, dur=dur, tied=tied, properties=properties)

    if noteType == 'unpitched':
        note.addSymbol(symbols.Notehead('x'))

    if context.fixSpelling:
        note.pitchSpelling = notename

    if notations:
        for notation in notations:
            if notation.skip:
                continue
            if notation.kind == 'articulation':
                articulation = scoring.definitions.normalizeArticulation(notation.value)
                if articulation:
                    note.addSymbol('articulation', articulation)
                else:
                    # If this is an unsupported articulation, at least save it as a property
                    logger.warning(f"Articulation not supported: {notation.value}")
                    note.properties['mxml/articulation'] = notation.value
            elif notation.kind == 'ornament':
                if notation.value == 'wavy-line':
                    kind = notation.properties['type']
                    key = _makeSpannerId('trilline', notation.properties, 'number')
                    if kind == 'start':
                        spanner = symbols.TrillLine(kind='start',
                                                    placement=notation.properties.get('placement', ''),
                                                    startmark=notation.properties.get('startmark', ''))
                        note.addSymbol(spanner)
                        context.spanners[key] = spanner
                    else:
                        assert kind == 'stop'
                        startspanner = context.spanners.pop(key)
                        startspanner.makeEndSpanner(note)
                elif notation.value == 'tremolo':
                    tremtype = notation.properties.get('tremtype', 'single')
                    nummarks = notation.properties.get('nummarks', 2)
                    note.addSymbol(symbols.Tremolo(tremtype=tremtype, nummarks=nummarks))
                else:
                    if ornament := scoring.definitions.normalizeOrnament(notation.value):
                        note.addSymbol('ornament', ornament)
                    else:
                        note.properties['mxml/ornament'] = notation.value
            elif notation.kind == 'fermata':
                note.addSymbol('fermata', scoring.definitions.normalizeFermata(notation.value))
            elif notation.kind == 'dynamic':
                dynamic = notation.value
                if dynamic2 := scoring.definitions.normalizeDynamic(dynamic):
                    note.dynamic = dynamic2
                else:
                    note.addText(dynamic, placement='below', fontstyle='italic,bold')
            elif notation.kind == 'fingering':
                note.addSymbol(symbols.Fingering(notation.value))
            elif notation.kind == 'notehead':
                note.addSymbol(symbols.Notehead(shape=notation.value))
            elif notation.kind == 'text':
                note.addSymbol(symbols.Text(notation.value,
                                            placement=notation.properties.get('placement', 'above'),
                                            fontstyle=notation.properties.get('fontstyle')))
            elif notation.kind == 'spanner':
                spannertype = notation.properties['type']
                key = _makeSpannerId(notation.value, notation.properties)
                if spannertype == 'start':
                    cls = notation.properties.pop('class')
                    spanner = cls(kind='start',
                                  linetype=notation.properties.get('line-type', 'solid'),
                                  color=notation.properties.get('color', ''))
                    if notation.properties:
                        for k, v in notation.properties.items():
                            spanner.setProperty(f'mxml/{k}', v)
                    context.spanners[key] = spanner
                    note.addSymbol(spanner)
                else:
                    startspanner = context.spanners.pop(key, None)
                    if not startspanner:
                        logger.error(f"No start spanner found for key {key}")
                    else:
                        startspanner.makeEndSpanner(note)

            elif notation.kind == 'bend':
                note.addSymbol(symbols.Bend(notation.properties['alter']))
    for symbol in notesymbols:
        note.addSymbol(symbol)

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
    if len(notes) == 1:
        return notes

    groups = []
    for note in notes:
        if note.properties.get('_chordCont'):
            groups[-1].append(note)
        else:
            groups.append([note])

    for i, group in enumerate(groups):
        if len(group) == 1:
            groups[i] = group[0]
        else:
            first = group[0]
            assert isinstance(first, Note)
            chord = first.asChord()
            for note in group[1:]:
                chord.append(note)
            chord.sort()
            chord.properties['voice'] = first.properties.get('voice', 1)
            groups[i] = chord
    return groups


def _measureDuration(beats: int, beattype: int) -> F:
    return F(beats*4, beattype)


@dataclass
class Direction:
    kind: str
    value: str = ''
    placement: str = ''
    properties: dict | None = None

    def getProperty(self, key, default=None):
        if not self.properties:
            return default
        return self.properties.get(key, default)


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


def _parseAttribs(attrib: dict, convertfuncs: dict[str, Callable]=None) -> dict:
    out = {}
    for k, v in attrib.items():
        convertfunc = None if not convertfuncs else convertfuncs.get(k)
        if v is not None:
            if convertfunc:
                v2 = convertfunc(v)
            elif v == 'yes':
                v2 = True
            elif v == 'no':
                v2 = False
            elif v.isnumeric():
                v2 = int(v)
            else:
                v2 = v
            out[k] = v2
    return out


def _applyDynamic(event: Note | Chord, dynamic: str) -> None:
    if dynamic2 := scoring.definitions.normalizeDynamic(dynamic):
        event.dynamic = dynamic2
    else:
        event.addText(dynamic, placement='below', fontstyle='italic')


def _parseTimesig(root: ET.Element) -> TimeSignature:
    """
     <time>
          <beats>3</beats>
          <beat-type>8</beat-type>
          <beats>2</beats>
          <beat-type>8</beat-type>
          <beats>3</beats>
          <beat-type>4</beat-type>
    """
    parts = []
    for item in root:
        if item.tag == 'beats':
            beats = item.text
            if '+' in beats:
                beatparts = tuple(int(p) for p in beats.split('+'))
                parts.append([beatparts, 4])
            else:
                parts.append([int(beats), 4])
        elif item.tag == 'beat-type':
            last = parts[-1]
            den = int(item.text)
            if isinstance(last[0], int):
                last[1] = den
            else:
                parts.pop()
                parts.extend([(p, den) for p in last[0]])
    if not parts:
        ET.dump(root)
        raise MusicxmlParseError("Could not find any time signature inside this element")
    return TimeSignature(*parts)


def _parseDirection(item: ET.Element, context: ParseContext) -> Direction | None:
    placement = item.attrib.get('placement')
    inner = item.find('direction-type')[0]
    tag = inner.tag
    if tag == 'dynamics':
        dynamic = inner[0].tag
        dynamic2 = scoring.definitions.normalizeDynamic(dynamic)
        if dynamic2:
            return Direction('dynamic', dynamic2)
        else:
            return Direction('words', dynamic, placement='below',
                             properties={'font-style': 'italic'})
    elif tag == 'words':
        # TODO: parse style / font / etc.
        if placement is None:
            placement = _parsePosition(inner)
        properties = _parseAttribs(inner.attrib, {'font-size': float})
        return Direction('words', inner.text, placement=placement,
                         properties=properties)
    elif tag == 'rehearsal':
        enclosure = inner.attrib.get('enclosure')
        return Direction('rehearsal', inner.text,
                         properties={'enclosure': enclosure,
                                     'placement': placement})
    elif tag == 'wedge':
        # a cresc or decresc hairpin
        wedgetype = inner.attrib['type']
        assert wedgetype in {'crescendo', 'diminuendo', 'stop'}
        properties = _parseAttribs(inner.attrib)
        return Direction('hairpin', wedgetype, placement=placement, properties=properties)
    elif tag == 'bracket':
        brackettype = inner.attrib['type']
        assert brackettype == 'start' or brackettype == 'stop', f"Expected 'start' or 'stop', got {brackettype}"
        properties = _parseAttribs(inner.attrib)
        return Direction('bracket', value=brackettype, placement=placement, properties=properties)
    elif tag == 'octave-shift':
        # In musicxml the pitch is always the sounding pitch. An octave shift shows the
        # sounding pitch at a different octave. 8va alta is then a down shift (-1),
        # 8va bassa is an up shift of 1 octave
        shifttype = inner.attrib['type']
        if shifttype != 'stop':
            shiftsize = inner.attrib.get('size', '8')
            numoctaves = {
                '8': 1,
                '15': 2
            }.get(shiftsize)
            if numoctaves is None:
                logger.error(f"Could not parse octave size: {shiftsize}")
                ET.dump(item)
                return
        else:
            numoctaves = 0
        if shifttype == 'up':
            numoctaves = -numoctaves
        assert shifttype in {'up', 'down', 'stop'}
        context.octaveshift = numoctaves
        return Direction('octaveshift', value=shifttype, properties={'octaves': numoctaves})
    elif tag == 'dashes':
        spannertype = inner.attrib['type']
        assert spannertype in {'start', 'stop'}
        return Direction('dashes', value=spannertype, placement=placement,
                         properties={k:v for k, v in inner.attrib.items()})


def _handleDirection(note: Note, direction: Direction, context: ParseContext):
    if direction.kind == 'dynamic':
        note.dynamic = direction.value
    elif direction.kind == 'words':
        note.addText(direction.value,
                     placement=direction.placement,
                     fontsize=direction.getProperty('font-size'),
                     fontstyle=direction.getProperty('font-style'))

    elif direction.kind == 'hairpin':
        if direction.value == 'crescendo':
            spanner = symbols.Hairpin(direction='<',
                                      niente=direction.getProperty('niente', False),
                                      placement=direction.placement,
                                      linetype=direction.getProperty('line-type', ''))
        elif direction.value == 'diminuendo':
            spanner = symbols.Hairpin(direction='>',
                                      niente=direction.getProperty('niente', False),
                                      placement=direction.placement,
                                      linetype=direction.getProperty('line-type', ''))
        else:
            spanner = symbols.Hairpin(direction='<', kind='end')
        note.addSymbol(spanner)

    elif direction.kind == 'bracket':
        key = _makeSpannerId('bracket', direction.properties)
        if direction.value == 'start':
            spanner = symbols.Bracket(kind='start',
                                      linetype=direction.getProperty('line-type', 'solid'),
                                      placement=direction.placement,
                                      text=direction.getProperty('text', ''))
            note.addSymbol(spanner)
            context.spanners[key] = spanner
        elif direction.value == 'stop':
            spanner = context.spanners.pop(key)
            spanner.makeEndSpanner(note)

    elif direction.kind == 'octaveshift':
        key = _makeSpannerId('octaveshift', direction.properties)
        if direction.value == 'up' or direction.value == 'down':
            octaves = direction.properties['octaves']
            spanner = symbols.OctaveShift(kind='start', octaves=octaves)
            context.spanners[key] = spanner
            note.addSymbol(spanner)
        else:
            assert direction.value == 'stop'
            startspanner = context.spanners.pop(key, None)
            if startspanner is None:
                logger.error(f"Open spanners: {context.spanners}")
                logger.error("Could not find matching octave shift")
                return
            endspanner = symbols.OctaveShift(kind='end', uuid=startspanner.uuid)
            note.addSymbol(endspanner)

    elif direction.kind == 'dashes':
        key = _makeSpannerId('dashes', direction.properties)
        if direction.value == 'start':
            spanner = symbols.LineSpan(kind='start', linetype='dashed')
            context.spanners[key] = spanner
            note.addSymbol(spanner)
        else:
            startspanner = context.spanners.pop(key, None)
            if not startspanner:
                logger.error(f"No start spanner found for key {key}")
            else:
                startspanner.makeEndSpanner(note)

    else:
        logger.warning(f"Direction not supported: {direction}")


def _parseAttributes(node: ET.Element, context: ParseContext):
    for item in node:
        if item.tag == 'divisions':
            context.divisions = int(item.text)
        elif item.tag == 'transpose':
            context.transposition = int(item.find('chromatic').text)


def _parsePart(part: ET.Element, context: ParseContext
               ) -> tuple[ScoreStruct, dict[int, Voice]]:
    """
    Parse a part

    Args:
        part: the <part> subtree

    Returns:
        a tuple (dict of voices, scorestruct)
    """
    # context = context.copy()
    sco = ScoreStruct()
    quarterTempo = 60
    beats, beattype = 4, 4
    subdivisions = None
    voices: dict[int, list[Note]] = {}
    measureCursor = F(0)
    directions: list[Direction] = []

    for measureidx, measure in enumerate(part.findall('measure')):
        cursor = measureCursor

        # we find first the metronome, if present, so that we can add a new
        # measure definition if there is a change in time-signature
        # NB: we do not parse <sound tempo=".."> tags
        if (metronome := measure.find('./direction/direction-type/metronome')) is not None:
            quarterTempo = _parseMetronome(metronome)

        measureProperties = {}
        if (time := measure.find('./attributes/time')) is not None:
            timesig = _parseTimesig(time)
            if len(timesig.parts) == 1:
                beats, beattype = timesig.raw
                subdivisions = None
            else:
                subdivisions = [num for num, den in timesig.normalizedParts]
                beats = sum(subdivisions)
                beattype = timesig.normalizedParts[0][1]
            if symbol := time.attrib.get('symbol'):
                measureProperties['symbol'] = symbol

        if (keynode := measure.find('./attributes/key')) is not None:
            fifths = int(keynode.find('fifths').text)
            mode = keynode.find('mode').text
            measureProperties['keySignature'] = (fifths, mode)

        sco.addMeasure(timesig=(beats, beattype), quarterTempo=quarterTempo,
                       subdivisions=subdivisions, **measureProperties)

        for item in measure:
            tag = item.tag
            if tag == 'attributes':
                _parseAttributes(item, context)
                #if (divisionsTag := item.find('divisions')) is not None:
                #    context.divisions = int(divisionsTag.text)
            elif tag == 'direction':
                if (direction := _parseDirection(item, context=context)) is not None:
                    directions.append(direction)

            elif tag == 'backup':
                dur = F(int(item.find('duration').text), context.divisions)
                cursor -= dur

            elif tag == 'forward':
                dur = F(int(item.find('duration').text), context.divisions)
                cursor += dur

            elif tag == 'note':
                note = _parseNote(item, context)
                voicenum = int(note.properties.get('voice', 1))
                cursorWithinMeasure = cursor - measureCursor
                note.offset = cursor
                for direction in directions:
                    if direction.kind == 'rehearsal':
                        box = direction.properties.get('enclosure')  # or 'square'
                        box = scoring.definitions.normalizeEnclosure(box)
                        if cursorWithinMeasure == 0:
                            sco.addRehearsalMark(measureidx, direction.value, box=box)
                        else:
                            # A rehearsal mark in the middle of the measure?? This
                            # can be added as a text annotation
                            note.addText(direction.value,
                                         placement='above',
                                         fontstyle='bold',
                                         box=box)
                    else:
                        _handleDirection(note, direction, context=context)

                voices.setdefault(voicenum, []).append(note)
                cursor += note.dur
                # each note consumes all directions until now
                directions.clear()

            elif tag == 'barline':
                location = item.attrib.get('location', 'right')
                # We do not support repeats
                if (barstylenode := item.find('bar-style')) is not None:
                    barstyle2 = scoring.definitions.normalizeBarstyle(barstylenode.text)
                    if not barstyle2:
                        logger.warning(f"Bartyle {barstylenode.text} unknown")
                        barstyle2 = 'single'
                    if location == 'right':
                        mdef = sco.getMeasureDef(measureidx)
                        mdef.setBarline(barstyle2)

        # end measure
        if measure.attrib.get('implicit') == 'yes':
            # A 'pickup' measure. Must not necessarilly be the first measure
            measureDur = sco.measuredefs[0].durationBeats()
            for voicenum, voice in voices.items():
                filledDur = cursor - measureCursor
                unfilledDur = measureDur - filledDur
                if unfilledDur > 0:
                    for item in voice:
                        if item.offset >= measureCursor:
                            item.offset += unfilledDur
                    voice.append(Rest(unfilledDur, offset=measureCursor))
                    voice.sort(key=lambda item: item.offset)
        measureCursor += F(beats*4, beattype)

    if len(sco) == 0:
        sco.addMeasure((4, 4), quarterTempo=60)

    activeVoices = {voicenum: Voice(_joinChords(notes))
                    for voicenum, notes in voices.items() if notes}

    # Convert glissando lines to actual glissando if the notes
    # involved are contiguous.
    # TODO: the current algorithm does not account for tied notes
    for voice in activeVoices.values():
        for event in voice.items:
            assert isinstance(event, (Note, Chord))
            if not event.symbols:
                continue
            for s in event.symbols:
                if isinstance(s, symbols.NoteheadLine) and s.kind == 'start':
                    anchor = s.anchor()
                    endanchor = s.partnerSpanner().anchor()
                    if voice.itemAfter(anchor) == endanchor:
                        anchor.gliss = True

    return sco, activeVoices

def _escapeString(s: str) -> str:
    return s.replace('"', r'\"')


def _guessEncoding(path: str, length=1024) -> str:
    import chardet
    teststr = open(path, "rb").read(length)
    info = chardet.detect(teststr)
    return info['encoding']


@dataclass
class PartDef:
    node: ET.Element = None
    partid: str = ''
    name: str = ''
    shortname: str = ''
    nameDisplay: str = ''
    shortnameDisplay: str = ''

    def __post_init__(self):
        if self.node is None:
            return
        self.partid = self.node.get('id')
        if not self.name and (name := self.node.find('part-name')) is not None:
            # Remove extraneous characters
            self.name = _escapeString(name.text)
        if not self.shortname and (shortname := self.node.find('part-abbreviation')) is not None:
            self.shortname = _escapeString(shortname.text)
        if not self.nameDisplay and (nameDisp := self.node.find('part-name-display')) is not None:
            self.nameDisplay = _escapeString(nameDisp.find('display-text').text)
        if not self.shortnameDisplay and (shortnameDisp := self.node.find('part-abbreviation-display')) is not None:
            self.shortnameDisplay = _escapeString(shortnameDisp.find('display-text').text)


def parseMusicxmlFile(path: str, fixSpelling=False) -> Score:
    """
    Read a musicxml file and parse its contents

    Args:
        path: the path to a musicxml file
        fixSpelling: if True, do not fix the enharmonic spelling to the
            note names read in the musicxml definition

    Returns:
        a Score

    """
    encoding = _guessEncoding(path)
    logger.debug(f"Opening musicxml file '{path}' with encoding: {encoding}")
    xmltext = open(path, "r", encoding=encoding).read()
    return parseMusicxml(xmltext, fixSpelling=fixSpelling)


def parseMusicxml(xml: str, fixSpelling=False) -> Score:
    """
    Parse the musicxml string

    The string should represent a valid musicxml document

    Args:
        xml: the musicxml string
        fixSpelling: if True, do not fix the enharmonic spelling to the
            note names read in the musicxml definition

    Returns:
        a Score

    """
    metadata = {}
    root = ET.fromstring(xml)
    if root.tag != 'score-partwise':
        raise MusicxmlParseError(f'Only score-partwise format is supported, got {root.tag}')
    version = root.get('version')
    if version:
        metadata['mxml/version'] = version

    # Parse metadata
    if (movementTitle := root.find('movement-title')) is not None:
        metadata['mxml/movement-title'] = movementTitle.text

    if (identification := root.find('identification')) is not None:
        if (creator := identification.find('creator')) is not None:
            creatortype = creator.get('type')
            metadata[f'mxml/identification/{creatortype}'] = creator.text

    # part list
    partlist = root.find('part-list')
    if partlist is None:
        raise MusicxmlParseError("No part-list tag found")

    partsRegistry: dict[str, PartDef] = {}
    for item in partlist:
        if item.tag == 'score-part':
            partdef = PartDef(item)
            assert partdef.partid
            partsRegistry[partdef.partid] = partdef

    # first, find the divisions. They must be defined in at least one part
    if (divisionsNode := root.find('./part/measure/attributes/divisions')) is None:
        logger.warning("This score does not define divisions. Using a default of 1")
        divisions = 1
    else:
        divisions = int(divisionsNode.text)

    rootcontext = ParseContext(divisions=divisions,
                               fixSpelling=fixSpelling)

    scorestructs: list[ScoreStruct] = []
    allvoices: list[Voice] = []
    for partidx, part in enumerate(root.findall('part')):
        context = rootcontext.copy()
        partid = part.get('id')
        logger.debug("Parsing part", partidx, partid)
        if partid is None:
            ET.dump(part)
            logger.error(f"Part definition does not have an id: {partidx}")
        scorestruct, voicesdict = _parsePart(part, context)
        if partid:
            partdef = partsRegistry.get(partid)
            if not partdef:
                logger.error(f"No corresponding part definition for id {partid}")
                partdef = PartDef(partid=partid, name=partid)
        else:
            partid = f'P{partidx}'
            partdef = PartDef(partid=partid, name=partid, shortname=partid)
        voicename = partdef.nameDisplay or partdef.name or partdef.partid
        shortname = partdef.shortnameDisplay or partdef.shortname or ''
        if len(voicesdict) == 1:
            # Only one voice
            voice = next(iter(voicesdict.values()))
            assert isinstance(voice, Voice)
            voice.label = voicename
            voice.shortname = shortname

        else:
            for voicenum, voice in voicesdict.items():
                voice.label = f'{voicename}/{voicenum}'
                if shortname:
                    voice.shortname = f'{shortname}/{voicenum}'

        for voice in voicesdict.values():
            props = {
                'mxml/part-name': partdef.name,
                'mxml/part-name-display': partdef.nameDisplay,
                'mxml/part-abbreviation': partdef.shortname,
                'mxml/part-abbreviation-display': partdef.shortnameDisplay
            }
            if voice.properties is None:
                voice.properties = props
            else:
                voice.properties.update(props)

        allvoices.extend(voicesdict.values())
        scorestructs.append(scorestruct)
    if not allvoices:
        logger.error("No voices found")

    scorestruct = max(scorestructs, key=lambda s: len(s))
    scorestruct.title = metadata.get('mxml/movement-title', '')
    scorestruct.composer = metadata.get('mxml/identification/composer', '')
    score = Score(voices=allvoices, scorestruct=scorestruct)
    if score.properties is None:
        score.properties = {}
    score.properties.update(metadata)
    return score