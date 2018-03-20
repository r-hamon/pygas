# ######### COPYRIGHT #########
#
# Copyright(c) 2018
# -----------------
#
# * Ronan Hamon r<lastname_AT_protonmail.com>
#
# Description
# -----------
#
# pygas is a python package that regroups tools to transform graphs into a
# collection of signals.
#
# Version
# -------
#
# * pygas version = 0.1
#
# Licence
# -------
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ######### COPYRIGHT #########
import os
from setuptools import setup, find_packages
import sys

NAME = 'pygas'
DESCRIPTION = 'Python implementation of the transformation from graph to signals, and back.'
URL = 'https://github.com/r-hamon/{}'.format(NAME)
AUTHOR = 'Ronan Hamon'
AUTHOR_EMAIL = 'rhamon@protonmail.con'
INSTALL_REQUIRES = ['matplotlib>=2.1',
                    'numpy>=1.14',
                    'scipy>=1.0',
                    'networkx>=1.0']
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: End Users/Desktop',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Natural Language :: English',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python :: 3']
PYTHON_REQUIRES = '>=3'
PROJECTS_URLS = {'Bug Reports': URL + '/issues',
                 'Source': URL}
KEYWORDS = 'networks, graphs, signals'


if sys.argv[-1] == 'setup.py':
    print("To install, run 'python setup.py install'\n")


def get_version():
    v_text = open('VERSION').read().strip()
    v_text_formted = '{"' + v_text.replace('\n', '","').replace(':', '":"')
    v_text_formted += '"}'
    v_dict = eval(v_text_formted)
    return v_dict[NAME]


def set_version(path, VERSION):
    filename = os.path.join(path, '__init__.py')
    buf = ""
    for line in open(filename, "rb"):
        if not line.decode("utf8").startswith("__version__ ="):
            buf += line.decode("utf8")
    f = open(filename, "wb")
    f.write(buf.encode("utf8"))
    f.write(('__version__ = "%s"\n' % VERSION).encode("utf8"))


def setup_package():
    """Setup function"""
    # set version
    VERSION = get_version()

    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
        long_description = f.read()

    mod_dir = NAME
    set_version(mod_dir, get_version())
    setup(name=NAME,
          version=VERSION,
          description=DESCRIPTION,
          long_description=long_description,
          url=URL,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          classifiers=CLASSIFIERS,
          keywords=KEYWORDS,
          packages=find_packages(exclude=['doc', 'tests']),
          install_requires=INSTALL_REQUIRES,
          python_requires=PYTHON_REQUIRES,
          projects_urls=PROJECTS_URLS)


if __name__ == "__main__":
    setup_package()
