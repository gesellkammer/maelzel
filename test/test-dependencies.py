import argparse
from maelzel.core import *
from maelzel.dependencies import checkDependencies

print("Checking dependencies")
print(checkDependencies(force=True, fix=True))

