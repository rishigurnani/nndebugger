# NetDebugger

## Quick Start
Run tutorial.ipynb

## Motivation
Imagine a program that did not pass along helpful descriptions of errors, let alone provide a notification that an error has occurred. How would one even begin to debug such a nightmare? Error-laden deep learning systems are susceptible to such “silent failures”. The reason for silent failure in deep learning is that any code representing a mathematically valid expression will fly, even if the coded expression is not mathematically logical. Further, the capacity of a deep learning model may be large enough to compensate for illogical code. Thus, the performance of the model may look reasonable enough that one does not re-examine the code for errors. To mitigate chances of falling into this trap, we have collected and encoded a list of common-sense checks that deep learning systems should pass. Several checks are borrowed from a well-written blog post[1] by Andrej Karpathy, Tesla’s Director of Artificial Intelligence, while some are original. The implementation of these checks is original work and is known as NetDebugger. Failure of any check results in an error along with a helpful error message. The presentation will start with an overview of deep learning theory to motivate the logic in NetDebugger and end with a hands-on NetDebugger tutorial involving PyTorch, RDKit, and polymer data (made publicly available by the Ramprasad Group at https://khazana.gatech.edu/). NetDebugger has saved me time in my research and, I hope, will be useful to the community.

[1] Karpathy A., A Recipe for Training Neural Networks, http://karpathy.github.io/2019/04/25/recipe/
