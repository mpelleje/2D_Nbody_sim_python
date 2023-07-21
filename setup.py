from setuptools import setup, Extension
import numpy as np

module = Extension(
    "sheet_unfolding.cpy_unfolding",
    sources=["sheet_unfolding/cpy_unfolding/cpy_unfolding.cpp", "sheet_unfolding/cpy_unfolding/unfolding.cpp"],
    include_dirs=["sheet_unfolding/cpy_unfolding", np.get_include()],
    extra_compile_args=['-std=c++11']
)

setup(
    name="sheet_unfolding",
    author="Jens Stücker",
    author_email="jstuecker@dipc.org",
    ext_modules=[module],
    packages=["sheet_unfolding", "sheet_unfolding/sim"],
    version="0.1.0"
)