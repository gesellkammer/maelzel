from __future__ import annotations
import constraint
from emlib.iterlib import pairwise, window
from emlib import misc, mathlib
from math import sqrt, inf
import bpf4 as bpf
import copy
import logging
import dataclasses
import time
from typing import Callable

logger = logging.getLogger("maelzel.timescale")


class Timedout(Exception):
    pass


default = {
    'values': [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5] + list(range(6, 40))
}


@dataclasses.dataclass
class Rating:
    name: str
    weight: float
    func: Callable
    exp: float = 1.0


@dataclasses.dataclass
class Solution:
    slots: list[float]
    score: float = 0.0
    data: dict | None = None


def gridDistance(grid: list[float], elem0: float, elem1: float, exact=True):
    """
    Return the distance in indexes between elem1 and elem0

    Args:
        grid: the grid to search in
        elem0: the first element
        elem1: the second element
        exact: if True, assume that elem0 and elem1 are present in seq, otherwise the nearest
            element in seq is used

    Returns:
        the distance in indexes between elem1 and elem0
    """
    if not exact:
        elem0 = misc.nearest_element(elem0, grid)
        elem1 = misc.nearest_element(elem1, grid)
    return grid.index(elem1) - grid.index(elem0)


def getSolutions(problem: constraint.Problem,
                 numSlots: int,
                 maxSolutions=0,
                 timeoutSearch=0.) -> list[list[int]]:
    """

    Args:
        problem: the problem to solve, after having added all constraints
        numSlots: the number of variables (slots) defined for this problem
        maxSolutions: stop searching for solutions if this amount is reached
        timeoutSearch: interrupt search after this period of time. Returns the
            results so far

    Returns:
        a list of solutions, where each solution is a list with the values for each slot

    Raises:
        Timedout: if timeout > 0 and no solution is found in that time
    """
    solutions = []
    t0 = time.time()
    for sol in problem.getSolutionIter():
        if not sol:
            continue
        vals = [sol[k] for k in range(numSlots)]
        solutions.append(vals)
        if maxSolutions and len(solutions) >= maxSolutions:
            break
        if timeoutSearch > 0 and time.time() - t0 > timeoutSearch:
            break
    return solutions


class Solver:
    def __init__(self, *, values=None, dur=None, absError=None, relError=None, timeout=0., fixedslots=None,
                 maxIndexJump=None, maxRepeats=None, maxSlotDelta=None, minSlotDelta=None,
                 monotonous='up', minvalue=-inf, maxvalue=inf):
        """
        Partition dur into timeslices
        To add extra constraints, use .addConstraint

        Args:
            values: possible values
            dur: sum of all values
            absError: absolute error of dur
            relError: relative error (only one of absError or relError should be given)
            timeout: timeout for the solve function, in seconds
            fixedslots: a dictionary of the form {0: 0.5, 2: 3} would specify that the
                slot 0 should have a value of 0.5 and the slot 2 a value of 3
            maxIndexJump: max. distance, in indices, between two slots
            maxRepeats: how many consecutive slots can have the same value
            maxSlotDelta: the max. difference between two slots
            minSlotDelta: the min. difference between two slots
            monotonous: possible values: 'up', 'down'. It indicates that all values
                should grow monotonously in the given direction
            minvalue: min. value for a slot
            maxvalue: max. value for a slot

        These are convenience values, we could just filter values (from param. `values`)
        which fall between these constraints (in fact this is what we do)
        """
        self.dur = dur
        self.values = values or default['values']
        self.absError = absError
        self.fixedslots = fixedslots
        self.timeout = timeout
        self.maxIndexJump = maxIndexJump
        self.maxRepeats = maxRepeats
        self.maxSlotDelta = maxSlotDelta
        self.minSlotDelta = minSlotDelta
        self.monotonous = monotonous
        self.relError = relError
        self._constraintCallbacks = []

        minvalue = minvalue if minvalue is not None else -inf
        maxvalue = maxvalue if maxvalue is not None else inf
        self.values = [v for v in self.values if minvalue <= v <= maxvalue]

    def copy(self):
        return copy.copy(self)

    def clone(self, **kws):
        out = self.copy()
        for key, val in kws.items():
            setattr(out, key, val)
        return out

    def solve(self, numslots: int):
        values = self.values
        dur = self.dur
        timeout = self.timeout
        problem = constraint.Problem()
        slots = list(range(numslots))
        problem.addVariables(slots, values)
        if dur is not None:
            if self.relError is None and self.absError is None:
                absError = min(values)
            elif self.absError is None:
                absError = self.relError * dur
            elif self.relError is None:
                absError = self.absError
            else:
                absError = min(self.absError, self.relError*dur)
            problem.addConstraint(constraint.MinSumConstraint(dur-absError))
            problem.addConstraint(constraint.MaxSumConstraint(dur+absError))

        if self.fixedslots:
            for idx, slotdur in self.fixedslots.items():
                try:
                    slot = slots[idx]
                    problem.addConstraint(lambda s, slotdur=slotdur: s==slotdur, variables=[slot])
                except IndexError:
                    pass

        self._applyConstraints(problem, slots)

        for callback in self._constraintCallbacks:
            callback(problem, slots)

        return getSolutions(problem, numSlots=numslots, timeoutSearch=timeout)

    def _applyConstraints(self, problem, slots):
        constr = problem.addConstraint
        if self.monotonous is not None:
            if self.monotonous == 'up':
                for s0, s1 in pairwise(slots):
                    constr(lambda s0, s1:  s0 <= s1, variables=[s0, s1])
            elif self.monotonous == 'down':
                for s0, s1 in pairwise(slots):
                    constr(lambda s0, s1:  s0 >= s1, variables=[s0, s1])
            else:
                raise ValueError("monotonous should be 'up' or 'down'")
        if self.minSlotDelta is not None:
            for s0, s1 in pairwise(slots):
                constr(lambda s0, s1: abs(s1 - s0) >= self.minSlotDelta, variables=[s0, s1])
        if self.maxIndexJump is not None:
            for s0, s1 in pairwise(slots):
                constr(lambda s0, s1: abs(gridDistance(self.values, s0, s1)) <= self.maxIndexJump, variables=[s0, s1])
        if self.maxRepeats is not None:
            for group in window(slots, self.maxRepeats + 1):
                constr(lambda *values: len(set(values)) > 1, variables=group)
        if self.maxSlotDelta is not None:
            for s0, s1 in pairwise(slots):
                constr(lambda s0, s1: abs(s1 - s0) <= self.maxSlotDelta, variables=[s0, s1])

    def addConstraint(self, slotIndexes, func) -> None:
        """
        Add a new constraint

        Args:
            slotIndexes: the indexes of the slots to take into account
            func: a function of the form (*slots) -> bool
                There should be a correspondence between the number of indexes
                passed as slotIndexes and the number of arguments expected by
                func

        Example:
            # Add a constraint to fix the initial value
            solver.addConstraint([0], lambda s0: s0 == myInitialValue)

            # Add a constraint so that every value is higher than the previous
            for i0, i1 in pairwise(range(numSlots)):
                solver.addConstraint([i0, i1], lambda s0, s1: s0 < s1)

        """
        def wrapped(problem, slots, indexes=slotIndexes):
            problem.addConstraint(func, variables=[slots[i] for i in indexes])
        self.addCallback(wrapped)

    def addCallback(self, func):
        """

        Args:
            func: a function of the form (problem, slots) -> None
                Given a Problem and a list of slots, this function should
                add any number of contraints to problem via problem.addConstraint

        Example:

            def growing(problem, slots):
                for s0, s1 in pairwise(slots):
                    problem.addConstraint(lambda s0, s1: s0 < s1, variables=[s0, s1])
            solver.addCallback(growing)
        """
        self._constraintCallbacks.append(func)
        return self


def asCurve(curve) -> bpf.BpfInterface:
    if isinstance(curve, (int, float)):
        return bpf.expon(0, 0, 1, 1, exp=curve)
    return bpf.asbpf(curve)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#             Extending a Solver
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _ascurve(curve) -> bpf.BpfInterface | None:
    if isinstance(curve, bpf.BpfInterface):
        return curve
    elif isinstance(curve, (int, float)):
        return bpf.expon(0, 0, 1, 1, exp=curve)
    elif curve is None:
        return None
    else:
        raise TypeError(f"curve should be a bpf, the exponent of a bpf, or None, got {curve}")


class Rater:

    def __init__(self,
                 relcurve: bpf.BpfInterface | float | None = None,
                 abscurve: bpf.BpfInterface | float | None = None,
                 varianceWeight=1.,
                 curveWeight=3.,
                 curveExp=1.):
        """
        relcurve: a bpf defined between x:0-1, y:0-1, or the exponent of an exponential curve
        """
        self.relcurve: bpf.BpfInterface | None = asCurve(relcurve) if relcurve is not None else None
        self.abscurve: bpf.BpfInterface | None = asCurve(abscurve) if abscurve is not None else None
        self.varianceWeight = varianceWeight
        self.curveWeight = curveWeight
        self.curveExp = curveExp
        self._ratings = []
        self._postinit()

    def _postinit(self):
        self.relcurve = asCurve(self.relcurve) if self.relcurve is not None else None
        self.abscurve = asCurve(self.abscurve) if self.abscurve is not None else None

    def __call__(self, solution: list[float]) -> Solution:
        numvalues = len(set(solution))
        ratedict = {
            'variance': (numvalues / len(solution), self.varianceWeight)
        }
        if self.relcurve is not None:
            relcurve = self.relcurve
        elif self.abscurve is not None:
            relcurve = (self.abscurve - solution[0]) / (solution[-1] - solution[0])
        else:
            relcurve = None

        if relcurve:
            score = rateRelativeCurve(solution, relcurve)
            ratedict['curve'] = (score**self.curveExp, self.curveWeight)

        for rating in self._ratings:
            score = rating.func(solution) ** rating.exp
            ratedict[rating.name] = (score, rating.weight)

        rates = list(ratedict.values())
        score = sqrt(sum((value**2)*weight for value, weight in rates) / sum(weight for _, weight in rates))
        return Solution(slots=solution, score=score, data=ratedict)

    def clone(self, **kws):
        out = copy.copy(self)
        for k, v in kws.items():
            setattr(out, k, v)
        out._postinit()
        return out

    def addRating(self, name, weight, func):
        """
        Example: rate higher solutions which have a small error

        NB: put extra info in the lambda itself

        rater.addRating("minError", weight=2,
                        func=lambda slots, dur=10, absError=2: abs(sum(slots)-dur)/absError)
        """
        self._ratings.append(Rating(name, weight, func))


def rateRelativeCurve(slots: list[float], relcurve: bpf.BpfInterface, plot=False) -> float:
    solxs = mathlib.linspace(0, 1, len(slots))
    x0 = min(slots)
    x1 = max(slots)
    if x0 == x1:
        diff = 1
    else:
        solys = [mathlib.linlin(slot, x0, x1,0, 1) for slot in slots]
        solcurve = bpf.core.Linear(solxs, solys)
        if plot:
            solcurve.plot(show=False)
            relcurve.plot(show=True)
        diff = (solcurve - relcurve).abs().integrate()
    assert diff <= 1, diff
    score = (1 - diff)
    return score


def solve(solver: Solver, numslots: int | list[int], rater: Rater=None,
          report=False, reportMaxRows=10) -> list[Solution]:
    """
    numslots: the number of slots to use, or a list of possible numslots

    Example
    ~~~~~~~

    values = [0.5, 1, 1.5, 2, 3, 5, 8]
    solver = Solver(values=values, dur=3, relError=0.1, monotonous='up')
    rater = Rater(relcurve=1.5)
    solutions = solve(solver=solver, numslots=4, rater=rater, report=True)
    best = solutions[0]

    """
    allsolutions = []
    possibleNumslots: list[int] = [numslots] if isinstance(numslots, int) else numslots
    for numslots in possibleNumslots:
        solutions = solver.solve(numslots)
        allsolutions.extend(solutions)
    ratedSolutions = []
    for sol in allsolutions:
        if rater is not None:
            sol = rater(sol)
        else:
            sol = Solution(sol, 0, None)
        ratedSolutions.append(sol)
    if rater is not None:
        ratedSolutions.sort(reverse=True, key=lambda solution: solution.score)
    if report:
        reportSolutions(ratedSolutions[:reportMaxRows])
    return ratedSolutions


def reportSolutions(solutions: list[Solution], plotbest=0, rater=None) -> None:
    """
    If given a rater, the solution will be plotted against the desired relcurve
    """
    if not solutions:
        raise ValueError("No solutions!")
    table = []
    for solution in solutions:
        ratings = solution.data.get('ratings') if solution.data else None
        if ratings:
            infostr = "\t".join([f"{key}: {value[0]:.3f}x{value[1]}={value[0]*value[1]:.3f}"
                                 for key, value in ratings.items()])
        else:
            infostr = ""
        row = (solution.slots, solution.score, infostr)
        table.append(row)
    misc.print_table(table, headers=('slots', 'score', 'infostr'))
    if plotbest and rater is not None and rater.relcurve is not None:
        for sol in solutions[:plotbest]:
            rateRelativeCurve(sol.slots, rater.relcurve)
