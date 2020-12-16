import bpf4 as bpf
from typing import Dict, Callable

_curves = {}   # type: Dict[str, Callable]


def criticalband(freq):
    # type: (float) -> float
    """
    Equivalent Rectangular Bandwidth as a function
    of Centre Frequency (in Hz)

    http://web.mit.edu/HST.723/www/ThemePapers/Masking/Moore95.pdf
    """
    moore95 = _curves.get('moore95')
    if not moore95:
        moore95 = bpf.linear(
            30, 30,
            100, 35,
            200, 45,
            300, 50,
            500, 70,
            1000, 110,
            2000, 220,
            3000, 500,
            10000, 900
        ).keep_skope()
        _curves['moore95'] = moore95
    band = moore95(freq)  # type: float
    return band