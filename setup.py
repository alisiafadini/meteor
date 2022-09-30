#coding: utf8

"""

Setup for METEOR

"""

from glob import glob

try:
	from setuptools import setup

except ImportError:
	from disutils.core import setup

    
setup(name='meteor',
      author='Alisia Fadini',
      packages=['meteor'],
      package_dir={'meteor': 'meteor'})
