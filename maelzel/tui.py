"""
Text User Interface

This package provides utilities for generating simple output / user interfaces for
for the terminal
"""
from __future__ import annotations
import textwrap
import rich.panel
import rich.console
import sys


def insideInteractiveTerminal() -> bool:
    """
    Are we running interactively inside a terminal?
    Returns:

    """
    return sys.stdout.isatty()


def panel(text: str, title='', subtitle='',
          width: int = 0, padding=(0, 1), titlealign='center',
          bordercolor='', margin=(0, 0), dedent=True
    ) -> None:
    """
    Print a panel to the console

    Args:
        text: the content of the panel
        title: am optional title
        subtitle: an optional subtitle
        width: the width of the panel, in chars
        padding: distance from panel to text
        titlealign: alignment of the title
        bordercolor: color of the border
        margin: margin from the terminal to the border
        dedent: dedent the text

    Example
    ~~~~~~~

    .. code-block:: python

        >>> from maelzel import tui
        >>> tui.panel("This is my text", "mytitle", padding=1, margin=1, titlealign='left')

        ╭─ mytitle ───────╮
        │                 │
        │ This is my text │
        │                 │
        ╰─────────────────╯

    """

    console = rich.console.Console()
    style = bordercolor or "none"
    if dedent:
        text = textwrap.dedent(text)
    p = rich.panel.Panel.fit(text, title=title, width=width or None, border_style=style,
                             padding=padding, subtitle=subtitle,
                             title_align=titlealign)
    margintop, marginbottom = (margin, margin) if isinstance(margin, int) else margin
    if margintop:
        print("\n"*margintop)
    console.print(p)
    if marginbottom:
        print("\n"*marginbottom)


def menu(options: list[str]) -> int | None:
    """
    A cli menu

    Will raise RuntimeError if not running inside an interactive terminal

    Args:
        options: a list of string options

    Returns:
        the index of the selection option, or None if no selection was made

    """
    if not insideInteractiveTerminal():
        raise RuntimeError("Not inside an interactive terminal")

    from simple_term_menu import TerminalMenu
    terminalMenu = TerminalMenu(options, multi_select=False)
    menuindex = terminalMenu.show()
    assert menuindex is None or isinstance(menuindex, int)
    return menuindex
