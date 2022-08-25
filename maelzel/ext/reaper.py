"""
Utilities to read regions and markers from REAPER's .RPP files
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Marker:
    num: int
    "The marker number"

    time: float
    "The time of the marker"

    descr: str
    "A description of the marker"


@dataclass
class Region:
    start: float
    "Start time of the region (in seconds)"

    end: float
    "End time of the region (in seconds)"

    id: int
    "Region id"

    label: str
    "Region label"


def getRegions(rppfile: str) -> List[Region]:
    """
    Extract the regions as a list of Region instances from a .RPP file

    Args:
        rppfile: the path to a reaper .RPP file

    Returns:
        a list of Regions, where each region is a dataclass holding the
        attributes `start, end, id, label`
    """
    f = open(rppfile)
    regions = []

    def isRegion(line):
        # expects that line has been stripped
        return line.startswith("MARKER") and line[-1] == '1'

    def parseRegion(line: str) -> Tuple[int, float, str]:
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
        if isRegion(line):
            regionid, start, label = parseRegion(line)
            region_started = True
            break
    for line in f:
        line = line.strip()
        if not isRegion(line):
            break
        regionid, time, label = parseRegion(line)
        if region_started:
            end = time
            regions.append(Region(start, end, regionid, label))
            region_started = False
        else:
            region_started = True
            start = time
    return regions


def getMarkers(rpp_file:str) -> List[Marker]:
    """
    Extract markers from a .rpp file

    Args:
        rppfile: the .rpp file to process

    Returns:
        a list of Markers, where each Marker is a dataclass with the attributes
        ``num``, time``, ``descr``

    A marker in reaper is a line with the form::

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


def writeMarkers(csvfile: str, markers) -> None:
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
