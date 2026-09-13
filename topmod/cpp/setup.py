from setuptools import setup, Extension
import pybind11
import os, sys

pb11_inc = pybind11.get_include()

ext = Extension(
    'topmod_core',
    sources=['dlfl_core.cpp'],
    include_dirs=[pb11_inc, '/usr/include/python3.12'],
    extra_compile_args=['-std=c++17', '-O3', '-march=native', '-fvisibility=hidden'],
    language='c++',
)

setup(name='topmod_core', ext_modules=[ext])
