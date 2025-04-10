import textwrap


_ENDOFBLOCK = "__ENDOFBLOCK__"


class IndentedWriter:
    """
    A class for writing text with indentation and line wrapping.

    Args:
        indentsize (int): The number of spaces to use for each level of indentation.
        maxwidth (int): The maximum width of each line.
        indents (int): The initial level of indentation.

    """
    def __init__(self, indentsize=2, maxwidth=92, indents=0):
        self.indentsize = indentsize
        self.maxwidth = maxwidth
        self.indents = indents
        self._blocks: list[list[str]] = []
        self._indentsPerBlock: list[int] = []

    def block(self, relativeindent=0) -> list[str]:
        """
        Create a new block with the specified relative indentation.

        Args:
            relativeindent (int): The relative indentation for the new block.

        Returns:
            list[str]: The new block.
        """
        block = []
        self._blocks.append(block)
        self.indents += relativeindent
        self._indentsPerBlock.append(self.indents)
        return block

    def currentBlock(self) -> tuple[int, list[str]]:
        """
        Get the current block and its indentation level.

        Returns:
            A tuple ``(indentation: int, block: list[str])`` where indentation is
            the indentation level of the current block and block is the current block.
        """
        if not self._blocks:
            block = self.block()
            return self._indentsPerBlock[-1], block
        else:
            return self._indentsPerBlock[-1], self._blocks[-1]

    def linewidth(self) -> int:
        """
        Get the current line width.

        Returns:
            int: The current line width.
        """
        indents, block = self.currentBlock()
        return sum(len(part) for part in block) + indents

    def add(self, text: str) -> None:
        """
        Add text to the current block.

        Args:
            text (str): The text to add.
        """
        indents, block = self.currentBlock()
        if block and block[-1] == _ENDOFBLOCK:
            block = self.block()
            indents = self.indents
        blockwidth = sum(len(part) for part in block) + indents
        if blockwidth > self.maxwidth * 0.4 and len(text) + blockwidth > self.maxwidth:
            block = self.block()
        block.append(text)

    def line(self, text: str) -> None:
        """
        Add a new line to the current block.

        Args:
            text (str): The text to add.
        """
        block = self.block()
        block.append(text)
        block.append(_ENDOFBLOCK)

    def join(self) -> str:
        """
        Join all blocks into a single string.

        Returns:
            str: The joined string.
        """
        lines = []
        for indents, block in zip(self._indentsPerBlock, self._blocks):
            if not block:
                continue
            if block[-1] == _ENDOFBLOCK:
                block.pop()
            blocktxt = textwrap.indent(' '.join(block), " " * (indents * self.indentsize))
            lines.append(blocktxt)
        return '\n'.join(lines)
