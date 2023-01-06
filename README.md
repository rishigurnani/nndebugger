# NetDebugger

## Installation
This repository is currently set up to run on 1) Mac OSX and 2) Linux/Windows machines with CUDA 10.2. Please raise a GitHub issue if you want to use this repo with a different configuration. Otherwise, please follow these steps for installation:

1. Install [poetry](https://python-poetry.org/) on your machine.
2. If Python3.7 is installed on your machine skip to step 3, if not you will need to install it. There are many ways to do this, one option is detailed below:
    * Install [Homebrew](https://brew.sh/) on your machine.
    * Run `brew install python@3.7`. Take note of the path to the python executable.
3. Clone this repo on your machine.
4. Open a terminal at the root directory of this repository.
5. Run `poetry env use /path/to/python3.7/executable`. If you installed Python3.7 with Homebrew, the path may be something like
  `/usr/local/Cellar/python\@3.7/3.7.13_1/bin/python3.7`.
7. Run `poetry install`.
8. If your machine is a Mac, run `poetry run poe torch-osx`. If not, run `poetry run poe torch-linux_win-cuda102`.
9. If your machine is a Mac, run `poetry run poe pyg-osx`. If not, run `poetry run poe pyg-linux_win-cuda102`.

## Example
Run tutorial.ipynb

## Motivation
Modified from my [talk](https://www.youtube.com/watch?v=TFWYoZoezrY) on neural network debugging: 

Imagine a program that did not pass along helpful descriptions of errors, let alone provide a notification that an error has occurred. How would one even begin to debug such a nightmare? Error-laden deep learning systems are susceptible to such “silent failures”. The reason for silent failure in deep learning is that a neural network is made up of many adaptive components; if one component is broken, the others may adapt such that the model's overall performance is reasonable. If we only look at performance, one would not be aware of errors in the broken module. To mitigate chances of falling into this trap, we have collected and encoded a list of common-sense checks that deep learning systems should pass. Several checks are borrowed from a well-written blog post<sup>1</sup> by Andrej Karpathy, Tesla’s Director of Artificial Intelligence, while some are original. The implementation of these checks is original work and is known as NetDebugger.

[1] Karpathy A., A Recipe for Training Neural Networks, http://karpathy.github.io/2019/04/25/recipe/

## Citation
If you use this repository in your work please consider citing us.
```
@misc{https://doi.org/10.48550/arxiv.2209.13557,
  doi = {10.48550/ARXIV.2209.13557},
  
  url = {https://arxiv.org/abs/2209.13557},
  
  author = {Gurnani, Rishi and Kuenneth, Christopher and Toland, Aubrey and Ramprasad, Rampi},
  
  keywords = {Materials Science (cond-mat.mtrl-sci), FOS: Physical sciences, FOS: Physical sciences},
  
  title = {Polymer informatics at-scale with multitask graph neural networks},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
## License
This repository is protected under a General Public Use License Agreement, the details of which can be found in `GT Open Source General Use License.pdf`.
