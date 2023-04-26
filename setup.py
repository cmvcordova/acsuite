#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Towards learning activty cliff aware molecular representations",
    author="César Miguel Valdez Córdova",
    author_email="cmvcordova@pm.me",
    url="https://github.com/user/project",
    install_requires=["pytorch-lightning", "hydra-core"], #add more
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
