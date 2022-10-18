from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"),
          encoding = "utf-8") as f:
    long_description = f.read()

setup(
    name = "pystats_utils",
    version = "0.1.0",
    description = "Statistics API",
    long_description = long_description,
    url = "",
    author = "Juan Antonio Marín Sanz",
    author_email = "b32masaj@gmail.com",
    license = "MIT",
    classifiers = [
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Statisticians",
        "Topic :: Statistics",
        "License :: MIT",
        "Programming Language :: Python :: 3",
    ],
    keywords = "pystats_utils statistics api",
    packages = find_packages(),
    package_dir = {"pystats_utils":"pystats_utils"}
)
