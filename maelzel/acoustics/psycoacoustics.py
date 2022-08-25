import bpf4 as bpf


# http://web.mit.edu/HST.723/www/ThemePapers/Masking/Moore95.pdf
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
).keep_slope()


def criticalband(freq: float) -> float:
    """
    Equivalent Rectangular Bandwidth as a function of Centre Frequency (in Hz)
    """
    return moore95(freq)
