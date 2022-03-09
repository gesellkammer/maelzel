#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import setup

readme = open('README.rst').read()
version = (0, 3, 2)

setup(
    name='maelzel',
    python_requires=">=3.9",
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
    ],
    include_package_data=True,
    install_requires=[
        "emlib>=1.2.0",
        "numpy",
        "scipy",
        "matplotlib",
        "music21",
        "bpf4",
        "configdict>=1.0.0",
        "appdirs",
        "tabulate",
        "sndfileio>=1.8.1",
        "pillow",
        "cachetools",
        "ctcsound",
        "numpyx",
        "watchdog",
        "python-constraint",
        "pyyaml",
        "rtmidi2",
        # "samplerate",   # https://pypi.org/project/samplerate/
        "resampy",
        "psutil",
        "csoundengine>=1.1",
        "pitchtools>=1.3",
        "lxml",
        "quicktions",
        "librosa"

    ],
    license="BSD",
    zip_safe=False,
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9'
    ]
)
