import rich.panel
import rich.console


def panel(text: str, title: str = None, width:int = None, padding=(0, 1),
          bordercolor: str = None, margin=(0, 0)) -> None:
    console = rich.console.Console()
    style = bordercolor or "none"
    p = rich.panel.Panel.fit(text, title=title, width=width, border_style=style,
                             padding=padding)
    margintop = 0
    marginbottom = 0
    if margin:
        margintop, marginbottom = (margin, margin) if isinstance(margin, int) else margin
        print("\n"*margintop)
    console.print(p)
    if margin:
        print("\n"*marginbottom)