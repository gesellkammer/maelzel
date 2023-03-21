#!/usr/bin/env python
import argparse
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument("--nolabel", action='store_true')
parser.add_argument("notebooks", nargs="+")

args = parser.parse_args()
addlabel = not args.nolabel

for notebook in args.notebooks:
    cmd = ["jupyter-nbconvert", "--to", "rst", notebook]
    subprocess.call(cmd)
    rst = os.path.splitext(notebook)[0] + '.rst'
    if not os.path.exists(rst):
        print(f"rst file {rst} not found!")
    txt = open(rst).read()
    txt = txt.replace('.. code:: ipython3', '.. code:: python')
    txt = txt.replace('0.9em', '0.8em')
    if addlabel:
        base = os.path.splitext(os.path.split(notebook)[1])[0]
        label = f'{base}_notebook'
        txt = f"\n.. _{label}:\n\n" + txt
    open(rst, "w").write(txt)
    
