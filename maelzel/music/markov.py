import random
from collections import deque
from emlib.distribute import dither_curve


def file2seq(path: str):
    for line in open(path):
        for word in line.rstrip().split():
            yield word


class ChainFixed:

    def __init__(self, seq, order):
        suffixmap = getmatrix(seq, order)
        self.suffixmap = suffixmap
        
    def generate(self, start=None):
        suffixmap = self.suffixmap
        # choose a random prefix (not weighted by frequency)
        cursor = start or random.choice(suffixmap.keys())
        while True:
            suffixes = suffixmap.get(cursor, None)
            if suffixes is None:
                # this node has no connections
                # if backtrack and len(cursor) > 1:
                break
            word = random.choice(suffixes)
            print("state: ", cursor, " possible jumps: ", suffixes, " selection: ", word)
            
            yield word
            cursor = cursor[1:] + (word,)


class ImpossibleStep(Exception): pass


class ChainVar:
    
    def __init__(self, seq, maxorder, minorder=1, constraint=None):
        """
        A markov chain with variable order
        seq: a sequence of states to build a chain
        """
        self.suffixmaps = {order:getmatrix(seq, order) for order in range(1, maxorder+1)}
        self.maxorder = maxorder
        self.minorder = minorder
        self.constraint = constraint
        self.history = deque([], 20)
        
    def step(self, previous=None):
        """
        Returns validstate, newstep

        If no possible step from the given state is possible, newstep will be None
        If the given state can't be reduced to a validstate (even by reducing the order), 
        then validstate will be None

        """
        state = self.validstate(previous)
        if state is None:
            return None, None
        order = len(state)
        suffixes = self.suffixmaps[order].get(state, None)
        if suffixes is None:
            return state, None
        if self.constraint is None:
            newstep = random.choice(suffixes)
        else:
            history = self.history if self.history else list(state)
            while True:
                newstep = random.choice(suffixes)
                history.append(newstep)
                if self.constraint(self.history):
                    break
                history.pop()
        self.history.append(newstep)
        return state, newstep

    def validstate(self, state=None):
        """
        Given a state of order between minorder and maxorder,
        if finds the highest order which matches the given state

        state can be None, in which case a valid state of the highest
        possible order will be returned

        Returns: state --> a seq. of states, len(state) == order

        NB: state will be None if the given state can't be reduced to a 
            valid node
        """
        if state is None:
            return random.choice(self.suffixmaps[self.maxorder].keys())
        elif not isinstance(state, tuple):
            raise ValueError("A state should be a tuple with values belonging to the chain")
        if len(state) > self.maxorder:
            state = state[-self.maxorder:]
        order = len(state)
        suffixmaps = self.suffixmaps
        while order >= self.minorder:
            if state in suffixmaps[order]:
                return state
            order -= 1
            state = state[1:]
        return None

    def generate(self, start=None, restart=False):
        state = self.validstate(start)
        while True:
            state, step = self.step(state)
            if step is None:
                if not restart:
                    break
                state = self.validstate()
                continue
            if self.constraint is not None and self.history:
                olditem = self.history.popleft()
                self.history.append(step)
                ok = self.constraint(self.history)
                if not ok:
                    self.history.appendleft(olditem)
                    continue
            yield step
            if len(state) == self.maxorder:
                state = state[1:]
            state += (step,)
            self.history.append(step)

    def __iter__(self):
        return self.generate()


def morph(chains, curve, numsteps, start=None, hook=None):
    """
    curve: a curve x->(any interval), y->(0, len(chains))
    chains: a seq. of 
    hook: if given, a function (stepnum, state) -> chainindex, will
          override the curve

    curve is used to index chains

    morph([ch1, ch2, ch3], bpf.linear(0, 0, 1, 2), 30)
    """
    chainidxs = dither_curve(curve, numsteps)
    out = []
    state = start
    for i, chainidx in enumerate(chainidxs):
        if hook:
            chainidx = hook(i, state)
        chain = chains[chainidx]
        state, step = chain.step(state)
        if step is None:
            # print("no valid step for this chain")
            state, step = chain.step(None)
            assert step is not None
        out.append(step)
        state += (step,)
    return out


def getmatrix(seq, order):
    prefix = ()
    prefix2suffixes = {}
    for item in seq:
        if len(prefix) < order:
            prefix += (item,)
        else:
            nodes = prefix2suffixes.get(prefix, [])
            nodes.append(item)
            prefix2suffixes[prefix] = nodes
            prefix = prefix[1:] + (item,)
    return prefix2suffixes


def morph_find_numsteps(chains, curve, sumvalue, bounds):
    """
    Given a seq. of chains producing values, calculate the number of steps needed to 
    arrive at a given value.

    chains: a seq. of ChainVars
    curve: a bpf of x: arbitrary range, y: index to the chains
    sumvalue: the value to approximate by summing the values of the morphed chains
    """
    b0 = bounds[0]
    stepsize = (bounds[1] - bounds[0]) / 4
    inf = float("inf")
    bestdiff = inf
    cache = {}

    def getsum(steps, N=1):
        return sum(sum(morph(chains, curve, steps)) for _ in xrange(N))/float(N)
    
    numsteps = None
    while stepsize > 0:
        bestdiff = inf
        for steps in range(bounds[0], bounds[1]+stepsize, stepsize):
            sumnow = cache.get(steps)
            if sumnow is None:
                sumnow = getsum(steps, 8)
                cache[steps] = sumnow
            diff = abs(sumnow - sumvalue)
            if diff < bestdiff:
                bestdiff = diff
                numsteps = steps
            elif bestdiff < inf and diff/bestdiff > 20:
                break
            # print(steps, bestdiff, diff)
        stepsize = int(stepsize / 2)
        bounds = (max(b0, numsteps - stepsize - 1), numsteps + stepsize)
    return numsteps
    
