import textwrap


_ENDBLOCK = "__ENDBLOCK__"


class IndentedWriter:
    """
    A class for writing text with indentation and line wrapping.

    Args:
        indentSize: The number of spaces to use for each level of indentation.
        maxWidth: The maximum width of each line.
        indents: The initial level of indentation.

    """
    def __init__(self, indentSize=2, maxWidth=92, indents=0):
        self.indentSize = indentSize
        self.maxWidth = maxWidth
        self.indents = indents
        self._blocks: list[list[str]] = []
        self._indentsPerBlock: list[int] = []
        self._stack: list[int] = []

    def __iadd__(self, num: int):
        self.indents += num

    def indent(self, num=1):
        """Can be used as a context manager

        w = IndentedWriter(...)
        with w.indent():
            ...
        # will dedent after exiting
        """
        self._stack.append(num)
        self.indents += num
        return self

    def dedent(self, num=1):
        for _ in range(num):
            indents = self._stack.pop()
            self.indents -= indents
        return self

    def __enter__(self):
        return

    def __exit__(self, *args, **kws):
        num = self._stack.pop()
        self.indents -= num

    def block(self, indents=0) -> list[str]:
        """
        Create a new block with the specified relative indentation.

        Args:
            indents: relative indentation for the new block.

        Returns:
            list[str]: The new block.
        """
        block = []
        self._blocks.append(block)
        self.indents += indents
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
        if block and block[-1] == _ENDBLOCK:
            block = self.block()
            indents = self.indents
        blockwidth = sum(len(part) for part in block) + indents
        if blockwidth > self.maxWidth * 0.4 and len(text) + blockwidth > self.maxWidth:
            block = self.block()
        block.append(text)

    def __call__(self, text: str) -> None:
        self.line(text)

    def line(self, text: str, indents=0, postindent=0) -> None:
        """
        Add a new line to the current block.

        Args:
            text: The text to add.
            indents: relative indentation for the line
            postindent: indentation after this line
        """
        block = self.block(indents=indents)
        block.append(text)
        block.append(_ENDBLOCK)
        if postindent:
            self.indents += postindent

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
            if block[-1] == _ENDBLOCK:
                block.pop()
            blocktxt = textwrap.indent(' '.join(block), " " * (indents * self.indentSize))
            lines.append(blocktxt)
        return '\n'.join(lines)
