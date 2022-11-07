def flattenGroup(group: DurationGroup) -> list[Notation]:
    out = []
    for item in group:
        if isinstance(item, Notation):
            out.append(item)
        else:
            out.extend(flattenGroup(item))
    return out


def _mergeAcrossBeats(groups: list[DurationGroup],
                      timesig: timesig_t,
                      quarterTempo: number_t,
                      minBeatFractionAcrossBeats=F(1),
                      mergedTupletsMaxDur=F(2),
                      allowedTupletsAcrossBeats=(1, 2, 3, 4, 8),
                      ) -> list[DurationGroup]:
    """
    After quantization of a measure, no group extends over the barrier of a beat.
    Here we merge notations across those boundaries for compatible
    duration groupTree.

    Args:
        groups: a list of duration groupTree, as returned by QuantizedMeasure.beatGroups
        timesig: the time-signature of the measure
        quarterTempo: the quarternote tempo of the measure
        minBeatFractionAcrossBeats:
        mergedTupletsMaxDur:
        allowedTupletsAcrossBeats:

    Returns:

    """
    # mark the beginning of each group. This is needed later on when rebuilding the
    # groupTree after merging across beats.
    def markGrouping(group: DurationGroup) -> None:
        for i, item in enumerate(group):
            if isinstance(item, Notation):
                if i == 0:
                    item.setProperty('.startGroup', True)
            else:
                markGrouping(item)

    for group in groups:
        markGrouping(group)

    flatNotations = sum((flattenGroup(group) for group in groups), [])
    merged = _mergeNotationsAcrossBeats(notations=flatNotations,
                                        timesig=timesig,
                                        minBeatFractionAcrossBeats=minBeatFractionAcrossBeats,
                                        allowedTupletsAcrossBeats=allowedTupletsAcrossBeats,
                                        mergedTupletsMaxDur=mergedTupletsMaxDur,
                                        quarterTempo=quarterTempo)
    for n in merged:
        print("... ", n)
    groups = _buildGroups(merged, timesig=timesig, mergedTupletsMaxDur=mergedTupletsMaxDur)
    return groups


def _mergeNotationsAcrossBeats(notations: list[Notation],
                               timesig: timesig_t,
                               minBeatFractionAcrossBeats=F(1),
                               allowedTupletsAcrossBeats=(1, 2, 3, 4, 8),
                               mergedTupletsMaxDur=F(2),
                               quarterTempo: number_t = 60,
                               ) -> list[Notation]:
    """
    Merge notations across beat boundaries

    Args:
        notations: the notations to try to merge
        timesig: the timesignature of the measure
        minBeatFractionAcrossBeats: min. fraction of the beat which can result in a merge. If 1, then a merge
            is only performed if the resulting merged Notation has a duration of at least 1 beat
        allowedTupletsAcrossBeats: which kind of tuplets are allowed across beats
        mergedTupletsMaxDur: the max. length (in Beats) of a merged tuplet
        quarterTempo: the tempo of the measure

    Returns:
        a list of merged notations. These notations are flat but the groupTree can be reconstructed based
        on their durRatios attributes (see _buildGroups)

    """
    mergedNotations = [notations[0]]
    beatDur = util.beatDurationForTimesig(timesig, quarterTempo)
    minMergedDuration = minBeatFractionAcrossBeats * beatDur
    for n1 in notations[1:]:
        n0 = mergedNotations[-1]
        assert isinstance(n0, Notation)
        mergedDur = n0.duration + n1.duration
        if n0.durRatios == n1.durRatios and core.notationsCanMerge(n0, n1):
            if ((mergedDur < minMergedDuration) or
                (n0.gliss and n0.duration+n1.duration >= 2 and n0.tiedPrev) or
                (n0.durRatios[-1].denominator not in allowedTupletsAcrossBeats) or
                (n0.durRatios[-1] != F(1, 1) and mergedDur > mergedTupletsMaxDur)):
                # can't merge
                mergedNotations.append(n1)
            else:
                merged = core.mergeNotations(n0, n1)
                merged.setProperty('.startGroup', False)
                mergedNotations[-1] = merged
        else:
            mergedNotations.append(n1)
    return mergedNotations


class GroupStack:
    def __init__(self):
        self.root = DurationGroup(durRatio=(1, 1))
        self.groups: list[DurationGroup] = [self.root]
        self.ratios: list[F] = [F(1)]

    def back(self, levels: int):
        if len(self.groups) < levels or levels < 1:
            print("*** Error")
            _dumpGroups([self.root], endline='------------------------------')
            raise ValueError(f"{levels=}, {self.ratios=}")
        self.groups[:] = self.groups[:-levels]
        self.ratios[:] = self.ratios[:-levels]

    def depth(self) -> int:
        return len(self.groups)

    def add(self, group: DurationGroup):
        self.groups[-1].items.append(group)
        self.groups.append(group)
        self.ratios.append(F(*group.durRatio))

    def __repr__(self):
        return f"GroupStack(ratios={self.ratios}, groupTree={self.groups})"


def _buildGroups(notations: list[Notation], timesig: timesig_t, mergedTupletsMaxDur: number_t
                 ) -> list[DurationGroup]:
    stack = GroupStack()

    def addToCurrentGroup(n: Notation, stack: GroupStack):
        # TODO: check if the group is too long
        group = stack.groups[-1]
        if group.durRatio != (1, 1) and group.duration() + n.duration > mergedTupletsMaxDur:
            print("group too long, splitting", group, n)
            stack.back(1)
            stack.add(DurationGroup(n.durRatios[-1]))
        stack.groups[-1].append(n)

    def commonPrefix(aratios: list[F], bratios: list[F]) -> list[F]:
        common = []
        for a, b in zip(aratios, bratios):
            if a == b:
                common.append(a)
            else:
                break
        return common

    for n in notations:
        common = commonPrefix(n.durRatios, stack.ratios[1:])
        backuplevels = len(stack.ratios[1:]) - len(common)
        forwardlevels = len(n.durRatios) - len(common)
        assert forwardlevels >= 0
        if backuplevels:
            stack.back(backuplevels)
        if forwardlevels:
            for ratio in n.durRatios[-forwardlevels:]:
                group = DurationGroup(ratio)
                stack.add(group)
        if backuplevels == 0 and forwardlevels == 0 and n.getProperty('.startGroup'):
            stack.back(1)
            group = stack.groups[-1]
            if group.durRatio != (1, 1) and len(n.durRatios) >= 2 and group.duration() + n.duration > 1:
                stack.back(1)
                stack.add(DurationGroup(n.durRatios[-2]))
            stack.add(DurationGroup(n.durRatios[-1]))
        addToCurrentGroup(n, stack)

    if stack.root.durRatio == (1, 1) and all(isinstance(x, DurationGroup) for x in stack.root.items):
        return stack.root.items
    return [stack.root]