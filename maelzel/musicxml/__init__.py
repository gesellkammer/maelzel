from __future__ import annotations
from .definitions import *


def pointsToMillimeters(points:float) -> float:
    return points * 7 / 20.


class LayoutUnitConverter:

    def __init__(self, millimeters, tenths=MUSICXML_TENTHS):
        """
        Musicxml measures distance in an abstract unit named "tenths".

        Args:
            millimeters: the number of millimeters corresponding to the given tenths
            tenths: the number of tenths corresponding to the given millimeters
        """
        self.millimeters = millimeters
        self.tenths = tenths

    def toTenths(self, mm:float) -> int:
        return int(mm / self.millimeters * self.tenths)

    def toMillimeters(self, tenths:int) -> float:
        return tenths / float(self.tenths) * self.millimeters

    @classmethod
    def fromStaffsize(cls, staffsizeInPoints) -> LayoutUnitConverter:
        mm = pointsToMillimeters(staffsizeInPoints)
        return cls(mm)


def microToAccidental(alter:float, semitoneDivisions=4) -> str:
    """
    Args:
        alter: XML `alter` element, a float indicating the midinote
            variation of the indicated step
            0 = no variation
            0.5 = quarter note up
        semitoneDivisions: the number of divisions per semitone
    """
    assert semitoneDivisions in (1, 2, 4)
    alter = round(alter*semitoneDivisions)/semitoneDivisions
    return MUSICXML_ACCIDENTALS[alter]

