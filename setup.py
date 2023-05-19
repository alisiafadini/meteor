

DESCRIPTION      =   "Scripts for crystallographic electron \
  density maps containing low occupancy species"
LONG_DESCRIPTION = """
METEOR contains commandline scripts and functions for finding \
  low occupancy species in crystallographic maps.
This module will be part of reciprocalspaceship and its "booster" package, rs-booster.
"""

try:
	from setuptools import setup, find_packages

except ImportError:
	from disutils.core import setup

    
setup(
    name='meteor',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author='Alisia Fadini',
    author_email="alisia.fadini15@imperial.ac.uk",
    install_requires=["reciprocalspaceship", "matplotlib", "seaborn"],
    packages=find_packages(),
    entry_points={
      "console_scripts": [
        "meteor.bckgr_subtract=meteor.background_subtraction.make_Nbgmax_map:main",
        "meteor.tv_denoise=meteor.TV_filtering.tv_denoise_map:main",
        "meteor.iterative_TV=meteor.TV_filtering.iterative_tv:main"
        ]
  }
)
