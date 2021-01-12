





class LayoutUnitConverter:

    def __init__(self, millimeters, tenths=MUSICXML_TENTHS):
        self.millimeters = millimeters
        self.tenths = tenths

    def to_tenths(self, mm:float) -> int:
        # tenths should always be an integer
        return int(mm / self.millimeters * self.tenths)

    def to_millimeters(self, tenths:int) -> float:
        # millimeters are floats
        return tenths / float(self.tenths) * self.millimeters

    @classmethod
    def from_staffsize(cls, staffsize_in_points):
        staffsize_in_mm = points_to_millimeters(staffsize_in_points)
        return cls(staffsize_in_mm)