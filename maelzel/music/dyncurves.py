from __future__ import division as _division
from ..iterlib import pairwise as _pairwise

silencio = '--'
ppppp = 'ppppp'
pppp = 'pppp'
ppp = 'ppp'
pp = 'pp'
p = 'p'
mf = 'mf'
f = 'f'
ff = 'ff'
fff = 'fff'
ffff = 'ffff'
fffff = 'fffff'

dyns = [silencio, ppppp, pppp, ppp, pp, p, mf, f, ff, fff, ffff, fffff]

c_pp = (
    (0, silencio),
    (30, ppppp),
    (50, ppp),
    (65, p),
    (70, mf)
)

c_ff = (
    (0, silencio),
    (10, ppp),
    (20, pp),
    (30, p),
    (40, mf),
    (50, ff),
    (60, fff),
    (65, ffff)
)

c_fp = (
    (0, silencio),
    (30, pppp),
    (40, ppp),
    (50, p),
    (55, f),
    (60, ff),
    (65, fff)
)

c_ppp = (
    (0, silencio),
    (40, ppppp),
    (50, pppp),
    (65, pp),
    (70, f)
)

c_fff = (
    (0, silencio),
    (20, p),
    (30, mf),
    (40, ff),
    (50, fff),
    (70, fffff)
)

c_pppf = (
    (0, silencio),
    (30, pppp),
    (50, pp),
    (57, p),
    (58, mf),
    (62, f),
    (66, fff)
)

c_pppfff = (
    (0, silencio),
    (30, pppp),
    (50, pp),
    (52, p),
    (53, f),
    (62, ff),
    (66, ffff)
)

dyncurves = {
    'ppp/fff' : c_pppfff,
    'fff/ppp' : c_pppfff,
    'ppp/f' : c_pppf,
    'f/ppp' : c_pppf,
    'fff' : c_fff,
    'c_ppp' : c_ppp,
    'ff/pp' : c_fp,
    'pp/ff' : c_fp,
    'ff'    : c_ff,
    'pp'    : c_pp,
    'ppp'   : c_ppp
}

def dbtodyn(db, curve):
    def get_enclosing(db, curve):
        if db <= (curve[0][0]):
            return curve[0], curve[0]
        if db >= (curve[-1][0]):
            return curve[-1], curve[-1]
        for (db0, dyn0), (db1, dyn1) in _pairwise(curve):
            if db0 <= db < db1:
                return (db0, dyn0), (db1, dyn1)
        raise WeShouldntBeHere
    def interpolate_index(db, db0, dyn0, db1, dyn1):
        index0 = dyns.index(dyn0)
        index1 = dyns.index(dyn1)
        if db0 == db1:
            return index0

        delta = (db - db0) / (db1 - db0)
        i = index0 + (index1 - index0) * delta
        i = int(i + 0.5)
        return i
    (db0, dyn0), (db1, dyn1) = get_enclosing(db, curve)
    index = interpolate_index(db, db0, dyn0, db1, dyn1)
    return dyns[index]

def _as_transformation(transformation):
    from bpf4 import bpf
    if isinstance(transformation, Number):
        constr = bpf.util.get_bpf_constructor("expon(%f)" % transformation)
    elif isinstance(transformation, basestring):
        constr = bpf,util.get_bpf_constructor(transformation)
    elif bpf.util.is_bpf(transformation):
        return transformation
    else:
        raise ValueError("transformation not understood. Either a bpf, an exponent or a string defining the bpf constructor")
    return constr(0, 0, 1, 1)
    
class DynTrans:
    def __init__(self, t0, curve0, t1, curve1, transformation='linear'):
        transformation = _as_transformation(transformation)
        self.transformation = transformation
        self.t0 = t0
        self.curve0 = curve0
        self.t1 = t1
        self.curve1 = curve1
    def __call__(self, db, t):
        return interpol_dyns(db, t, self.t0, self.curve0, self.t1, self.curve1, self.transformation)

class DynCurve:
    def __init__(self, curve):
        self.curve = curve
    def __call__(self, db):
        return dbtodyn(db, self.curve)

def interpol_dyns(db, t, t0, curve0, t1, curve1, transformation=None):
    dyn0 = dbtodyn(db, curve0)
    dyn1 = dbtodyn(db, curve1)
    index0 = dyns.index(dyn0)
    index1 = dyns.index(dyn1)
    delta = (t - t0) / (t1 - t0)
    if transformation is not None:
        delta = transformation(delta)
    i = index0 + (index1 - index0) * delta
    i = int(i + 0.5)
    return dyns[i]

def pp_to_ff(dur, transf='expon(2)'):
    transf = _as_transformation(transf)
    return DynTrans(0, c_pp, dur, c_ff, transf)
    
def ff_to_pp(dur, transf='expon(2)'):
    transf = _as_transformation(transf)
    return DynTrans(0, c_ff, dur, c_pp, transf)

ff = DynCurve(c_ff)
pp = DynCurve(c_pp)
fp = DynCurve(c_fp)