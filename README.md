# METEOR
Map Enhancement Tools for Ephemeral Occupancy Refinement
----------

[![Pytest](https://github.com/alisiafadini/meteor/actions/workflows/tests.yml/badge.svg)](https://github.com/your_username/your_repo/actions/workflows/tests.yml)
[![Mypy](https://github.com/alisiafadini/meteor/actions/workflows/mypy.yml/badge.svg)](https://github.com/your_username/your_repo/actions/workflows/mypy.yml)
[![Ruff](https://github.com/alisiafadini/meteor/actions/workflows/lint.yml/badge.svg)](https://github.com/your_username/your_repo/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/alisiafadini/meteor/graph/badge.svg?token=NCYMP06MNS)](https://codecov.io/gh/alisiafadini/meteor)

`meteor` is a tool for computing crystallographic difference maps. 

`meteor` specializes the robust identification of weak signals arising from minor but scientifically interesting populations, such as bound ligands or changes that occur in time-resolved experiments. That said, if you need an difference map, `meteor` can do it!


## quickstart

❗ `meteor` is currently **in beta**. We re-wrote everything recently, moving from a research code to something that can be robustly used as a tool. If you are willing to put up with a few sharp edges, it would be great if you give it a spin and then send us feedback: on how easy/not it was to use and what kinds of scientific results you obtain.

Finally: a word of caution. Expect changes in the coming weeks as we stress test the code. You might want to consider this before publishing any results with `meteor` until we exit `beta`. 

In that vein, for now please install the current master branch as per the following instructions. We'll have a sane
first stable version soon, which will be deployed straight to `PyPI` and `conda-forge`. Stay posted.

First, we recommend you make a new environment. For conda users,
```
conda create --name meteor python==3.11 --yes
conda activate meteor
```

Then install from github,
```
pip install git+https://github.com/alisiafadini/meteor
```

Once installed, you will have two command-line scripts. Ask for more info using `-h`:
```
meteor.diffmap -h
meteor.phaseboost -h
```
which compute denoised difference maps using the constant-phase approximation _vs._ iterative phase retrieval, respectively.


## philosophy: better science through automation

`meteor` aims to:

1. maximize signal to noise
2. be objective and reproducible (minimize user choice & bias)
3. be easy to use

Aim 1 is met using structure factor amplitude weighting (e.g. k-weighting, existing art) and TV denoising (new in the context of crystallography). Aims 2 and 3 are met through automatically setting parameters using negentropy maximization. For all the details, see our paper (coming soon to a preprint server near you).


## isomorphous data, please

METEOR is only for isomorphous difference maps, meaning the lattices/symmetries of the `native` and `derivative` datasets are comparable. If you need to compare non-isomorphous lattices, check out [matchmaps](https://github.com/rs-station/matchmaps).


