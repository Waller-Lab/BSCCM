"""The setup script."""

import setuptools
from os import path

with open("README.md", "r") as fh:
    long_description = fh.read()

# extract version
path = path.realpath("bsccm/_version.py")
version_ns = {}
with open(path, encoding="utf8") as f:
    exec(f.read(), {}, version_ns)
version = version_ns["__version__"]

setuptools.setup(
    name="bsccm",
    version=version,
    author="Henry Pinkard",
    author_email="henry.pinkard@gmail.com",
    description="Access the Berkeley Single Cell Computational Microscopy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/henrypinkard/BSCCM",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "zarr",
        "requests", 
        "tqdm"
    ],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)