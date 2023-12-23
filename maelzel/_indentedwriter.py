import textwrap


_ENDOFBLOCK = "__ENDOFBLOCK__"


class IndentedWriter:
    def __init__(self, indentsize=2, maxwidth=92, indents=0):
        self.indentsize = indentsize
        self.maxwidth = maxwidth
        self.indents = indents
        self._blocks: list[list[str]] = []
        self._indentsPerBlock: list[int] = []

    def block(self, relativeindent=0) -> list[str]:
        block = []
        self._blocks.append(block)
        self.indents += relativeindent
        self._indentsPerBlock.append(self.indents)
        return block

    def currentBlock(self) -> tuple[int, list[str]]:
        if not self._blocks:
            block = self.block()
            return self._indentsPerBlock[-1], block
        else:
            return self._indentsPerBlock[-1], self._blocks[-1]

    def linewidth(self) -> int:
        indents, block = self.currentBlock()
        return sum(len(part) for part in block) + indents

    def add(self, text: str) -> None:
        indents, block = self.currentBlock()
        if block and block[-1] == _ENDOFBLOCK:
            block = self.block()
            indents = self.indents
        blockwidth = sum(len(part) for part in block) + indents
        if blockwidth > self.maxwidth * 0.4 and len(text) + blockwidth > self.maxwidth:
            block = self.block()
        block.append(text)

    def line(self, text: str) -> None:
        block = self.block()
        block.append(text)
        block.append(_ENDOFBLOCK)

    def join(self) -> str:
        lines = []
        for indents, block in zip(self._indentsPerBlock, self._blocks):
            if not block:
                continue
            if block[-1] == _ENDOFBLOCK:
                block.pop()
            blocktxt = textwrap.indent(' '.join(block), " " * (indents * self.indentsize))
            lines.append(blocktxt)
        return '\n'.join(lines)
