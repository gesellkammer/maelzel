

def squarelattice_bandgapfreq(alpha:float, c=343.) -> float:
    """
    find the bang-gap frequency for a square lattice
    of distance alpha between each cylinder

    The central frequency of the band gap fBG is de-
    termined by the lattice constant α which presents
    the distance between  adjacent scatterers for a square
    lattice

    http://acoustique.ec-lyon.fr/publi/koussa_acta13.pdf
    """
    f = c/(2*alpha)
    return f
