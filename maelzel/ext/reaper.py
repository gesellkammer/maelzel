"""
Utilities to read regions and markers from REAPER's .RPP files
"""

from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Marker:
    num: int
    time: float
    descr: str


@dataclass
class Region:
    start: float
    end: float
    id: int
    label: str



def get_regions(rpp_file: str) -> List[Region]:
    """
    given a REAPER .rpp, extract the regions as a list of Region instances
    (start, end, id, label)
    """
    f = open(rpp_file)
    regions = []

    def is_region(line):
        # expects that line has been stripped
        return line.startswith("MARKER") and line[-1] == '1'

    def parse_region(line: str) -> Tuple[int, float, str]:
        words = line.split()
        # MARKER = words[0]
        regionid = int(words[1])
        time = float(words[2])
        # track = int(words[-1])
        label = " ".join(words[3:-1])
        label = label.replace('"', '')
        return regionid, time, label

    # skip until we find markers
    region_started = False
    start = 0
    for line in f:
        line = line.strip()
        if is_region(line):
            regionid, start, label = parse_region(line)
            region_started = True
            break
    for line in f:
        line = line.strip()
        if not is_region(line):
            break
        regionid, time, label = parse_region(line)
        if region_started:
            end = time
            regions.append(Region(start, end, regionid, label))
            region_started = False
        else:
            region_started = True
            start = time
    return regions


def get_markers(rpp_file:str) -> List[Marker]:
    """
    given a REAPER .rpp, extract the markers as a list of Markers
    (start, end, id, label)

    A marker in reaper is a line with the form

    MARKER 17  4.55480106957674 6560  0 0 1 B {55F210FA-D4E4-128E-514D-C013EAD05B78}
    const  num t                descr ? ? ? ?  uuid
    """
    f = open(rpp_file)

    def is_marker(line):
        # expects that line has been stripped
        return line.startswith("MARKER")

    def parse_marker(line: str) -> Marker:
        words = line.split()
        labelType = words[0]
        assert labelType == "MARKER"
        num = int(words[1])
        t = float(words[2])
        descr = words[3]
        return Marker(num=num, time=t, descr=descr)

    markers = []
    for line in f:
        line = line.strip()
        if is_marker(line):
            markers.append(parse_marker(line))
    return markers


def write_markers(csvfile: str, markers):
    """
    Writes a list of markers to disk as a .csv file.

    Reaper exchanges markers and regions as .csv with format

    id, Name, Start, End, Length, Color

    Args:
        csvfile: the file to write to
        markers: the markers to write. A list of tuples (name, start, [end])
            If an end is given, a Region is created

    start and end can be either a floating point, in which case
    they are interpreted as absolute time, or a string of the type
    "MM.BB.XXX" with MM=measure, BB=beat, XXX=subdivision    

    Notes about reaper's format:
    * id: M1, M2, R1, etc, where Mxx identifies a Marker and Rxx indentifies a Region

    """
    import csv
    markerid = 1
    regionid = 1
    rows = []
    for marker in markers:
        if len(marker) == 2:
            name, start = marker
            markerId = "M" + str(markerid)
            markerid += 1
            end = ""
            length = ""
            color = ""
        elif len(marker) == 3:
            name, start, end = marker
            length = ""
            color = ""
            markerId = "R" + str(regionid)
            regionid += 1
        else:
            raise ValueError(f"A marker should be a tuple of length 2 or 3, got {marker}")
        row = (markerId, name, start, end, length, color)
        rows.append(row)
    with open(csvfile, "w") as fileobj:
        writer = csv.writer(fileobj)
        writer.writerow("# Name Start End Length Color".split())
        for row in rows:
            writer.writerow(row)
