from pathlib import Path

def findRoot():
    p = Path(__file__).parent
    if (p/"index.rst").exists():
        return p.parent
    if (p/"setup.py").exists():
        return p
    if (p.parent/"setup.py").exists():
        return p.parent
    raise RuntimeError("Could not locate the root folder")


root = findRoot()
docs = root / "docs"
assert docs.exists()


from maelzel.core import *
cfg = getConfig()
rst = cfg.generateRstDocumentation(linkPrefix='config_')
outfile = docs / "configkeys.rst"
open(outfile, "w").write(rst)
