#!/usr/bin/env python3
#
# Copyright (C) 2020 Chi-kwan Chan
# Copyright (C) 2020 Steward Observatory
#
# This file is part of XAJ.
#
# XAJ is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# XAJ is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with XAJ.  If not, see <http://www.gnu.org/licenses/>.


from setuptools import setup, find_packages


setup(
    name='xaj',
    version='0.1.0',
    url='https://github.com/adxsrc/xaj',
    author='Chi-kwan Chan',
    author_email='chanc@arizona.edu',
    description='General relativistic ray tracing in python based on JAX',
    packages=find_packages('mod'),
    package_dir={'': 'mod'},
    python_requires='>=3.7',
    install_requires=[
        'jaxlib',
        'jax>=0.2.26',
        'tqdm',
    ],
)
