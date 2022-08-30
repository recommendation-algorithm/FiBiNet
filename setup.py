#! coding: utf-8
from setuptools import setup

setup(
    name='fibinet',
    version='1.0',
    description='Support FiBiNet and FiBiNet++',
    author='',
    author_email='',
    packages=['fibinet'],
    install_requires=["tensorflow==1.14", "pydot", "graphviz", "numpy", "pandas", "scikit-learn"],
)
