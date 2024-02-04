#!/usr/bin/env python
import os
from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="acsuite",
    py_modules = ['acsuite'],
    version="0.1",
    description="Provides basic functionality to learn and evaluate representations from activity cliff data",
    author="César Miguel Valdez Córdova",
    author_email="cmvcordova@pm.me",
    url="https://github.com/cmvcordova/acsuite",
    install_requires=requirements,
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train = src.train:main",
            "eval = src.eval:main",
        ]
    },
)
