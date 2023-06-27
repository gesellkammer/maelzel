"""
Text User Interface

This package provides utilities for generating simple output / user interfaces for
for the terminal
"""
from __future__ import annotations

import textwrap

import rich.panel
import rich.console


def panel(text: str, title: str = None, subtitle: str = None,
          width: int = None, padding=(0, 1), titlealign='center',
          bordercolor: str = None, margin=(0, 0), dedent=True
    ) -> None:
    console = rich.console.Console()
    style = bordercolor or "none"
    if dedent:
        text = textwrap.dedent(text)
    p = rich.panel.Panel.fit(text, title=title, width=width, border_style=style,
                             padding=padding, subtitle=subtitle,
                             title_align=titlealign)
    margintop = 0
    marginbottom = 0
    if margin:
        margintop, marginbottom = (margin, margin) if isinstance(margin, int) else margin
        print("\n"*margintop)
    console.print(p)
    if margin:
        print("\n"*marginbottom)


def menu(options: list[str]) -> int | None:
    from simple_term_menu import TerminalMenu
    terminalMenu = TerminalMenu(options)
    menuindex = terminalMenu.show()
    return menuindex
