#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob

from setuptools import setup

readme = open('README.rst').read()
version = (0, 9, 1)


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('maelzel/data')
print(extra_files)
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
        'maelzel.musicxml',
        'maelzel.transcribe'
    ],
    install_requires=[
        "emlib>=1.7.3",
        "numpy",
        "scipy",
        "matplotlib",
        "music21",
        "appdirs",
        "tabulate",
        "ctcsound",
        "cachetools",
        "pyyaml",
        "watchdog",
        "psutil",
        "lxml",
        "quicktions",
        "rich",
        "chardet",
        "simple-term-menu",
        "visvalingamwyatt",
        # "resampy",  # Do not include it by default, since this pulls numba

        "bpf4>=1.8.4",
        "configdict>=2.6",
        "sndfileio>=1.8.1",
        "numpyx>=1.3.1",
        "python-constraint",
        "csoundengine>=1.18.1",
        "pitchtools>=1.9.2",
        "risset>=2.1.0",
        "loristrck>=1.5.2",
        "vamphost>=1.2.1"
    ],
    license="LGPLv2",
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Multimedia :: Sound/Audio'
    ],
    include_package_data=True,
    package_data={'maelzel': extra_files},
)

