#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from setuptools import setup

readme = open('README.rst').read()
version = (0, 13, 0)


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('maelzel/data')

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
        'maelzel.musicxml',
        'maelzel.transcribe',
        'maelzel.partialtracking'
    ],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "appdirs",
        "tabulate",
        "cachetools",
        "pyyaml",
        "psutil",
        "lxml",
        "quicktions",
        "rich",
        "chardet",
        "simple-term-menu",
        "visvalingamwyatt",
        "distro",
        # "resampy",  # Do not include it by default, since this pulls numba
        "python-constraint",

        "emlib>=1.11.2",
        "ctcsound7>=0.3.0",
        "bpf4>=1.8.4",
        "configdict>=2.7.0",
        "sndfileio>=1.9.1",
        "numpyx>=1.3.1",
        "csoundengine>=1.22.0",
        "pitchtools>=1.11.0",
        "risset>=2.4.1",
        "loristrck>=1.5.6",
        "vamphost>=1.2.1",
        "lilyponddist>=0.3.0"
    ],
    license="LGPLv2",
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Multimedia :: Sound/Audio'
    ],
    include_package_data=True,
    package_data={'maelzel': extra_files},
)

