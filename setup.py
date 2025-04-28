#! /usr/bin/env python

from setuptools import setup, find_packages
import subprocess

# Set the package release version
major = 0
minor = 0
patch = 0

# Set the package details
name = 'DIABETES'
version = '.'.join(str(value)for value in (major, minor, patch))
author = 'Luis Zanon' 
url = 'https://github.com/luiszanro/GASTRIC-CANCER'
description = (
    'Project of Data Science'
    + 'DIABETES.'
)

# Now is commented I will integrate it later when creating the virtual env
# Also I will add the requirements for easy execution 
#with open('requirements.txt') as open_file:
#    install_requires = open_file.read()

# Set name of package and sub-packages

# Setup
setup(
    name=name,
    author=author,
    version=version,
    url=url,
    description=description   
)