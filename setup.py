#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import setup

readme = open('README.rst').read()
version = (0, 6, 0)

setup(
    name='maelzel',
    python_requires=">=3.10",
    version=".".join(map(str, version)),
    description='Utilities for sound, music notation, acoustics, etc',
    long_description=readme,
    author='Eduardo Moguillansky',
    author_email='eduardo.moguillansky@gmail.com',
    url='https://github.com/gesellkammer/maelzel',
    packages=[
        'maelzel',
        'maelzel.core',
        'maelzel.midi',
        'maelzel.snd',
        'maelzel.acoustics',
        'maelzel.ext',
        'maelzel.music',
        'maelzel.scoring',
        'maelzel.musicxml'
    ],
    include_package_data=True,
    install_requires=[
        "emlib>=1.7.3",
        "numpy",
        "scipy",
        "matplotlib",
        "music21",
        "bpf4",
        "configdict>=2.5",
        "appdirs",
        "tabulate",
        "sndfileio>=1.8.1",
        "pillow",
        "cachetools",
        "ctcsound",
        "numpyx>=1.2.0",
        "watchdog",
        "python-constraint",
        "pyyaml",
        "rtmidi2",
        "resampy",
        "psutil",
        "csoundengine>=1.13.4",
        "pitchtools>=1.9",
        "lxml",
        "quicktions",
        "rich",
        "risset",
        "chardet",
        "simple-term-menu",
        "visvalingamwyatt"
    ],
    license="BSD",
    zip_safe=False,
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9'
    ],
    package_data={'': ['data/*']},
)
