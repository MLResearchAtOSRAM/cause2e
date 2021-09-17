[![Build Status](https://dev.azure.com/OsramDS/cause2e_github/_apis/build/status/MLResearchAtOSRAM.cause2e?branchName=main)](https://dev.azure.com/OsramDS/cause2e_github/_build/latest?definitionId=24&branchName=main)
![Coverage](https://img.shields.io/azure-devops/coverage/OsramDS/cause2e_github/24)

## Getting started:
The easiest way of learning about cause2e's functionality and starting your own causal analyses is to check out this [example notebook](https://github.com/MLResearchAtOSRAM/cause2e/blob/main/examples/minimal_end_to_end_causal_analysis.ipynb), which can be easily adapted to fit the needs of your custom analysis. Additional notebooks with examples of more specific functionality are also provided.

## Overview:
The cause2e package provides tools for performing an **end-to-end causal analysis** of your data. If you have data and domain knowledge about the data generating process, it allows you to:
- learn a graphical causal model of the data generating process
- identify a statistical estimand for the causal effect that one variable has on another variable
- estimate the effect with various statistical techniques
- check the robustness of your results with respect to changes in the causal model

For **analyzing the whole system at once** after learning the causal graph, you can use a single command to
- visualize your qualitative domain knowledge
- estimate all possible direct, indirect and overall causal effects between your variables via do-calculus-backed linear regression
- receive a ranking of the strongest causal effects
- visualize all causal effects in heatmaps
- validate your model against a priori known quantitative causal effects
- receive a pdf report containing all relevant information

The main contribution of cause2e is the integration of two established causal packages that have currently been separated and cumbersome to combine:
- **Causal discovery methods** from the *py-causal* package [1], which is a Python wrapper around parts of the Java *TETRAD* software. It provides many algorithms for learning the causal graph from data and domain knowledge.

- **Causal reasoning methods** from the *DoWhy* package [2], which is the current standard for the steps of a causal analysis starting from a known causal graph and data:
    - Algebraically identifying a statistical estimand for a causal effect from the causal graph via do-calculus.
    - Using statistical estimators to actually estimate the causal effect.
    - Performing robustness tests to check how sensitive the estimate is to model misspecification and other errors.


## Structured API:
cause2e provides an **easy to use API** for performing an end-to-end causal analysis without having to worry about fitting together different libraries and data structures for causal discovery and causal reasoning:
- The **StructureLearner** class for causal discovery can
    - read and preprocess data
    - accept domain knowledge in a simple data format
    - learn the causal graph using *py-causal* algorithms
    - visualize the causal graph and the influence of the specified domain knowledge on the result
    - manually postprocess the resulting graph in case you want to add, delete or reverse some edges
    - check if the graph is acyclic and respects the domain knowledge
    - save the graph to various file formats

- The **Estimator** class for causal reasoning can
    - read data and imitate the preprocessing steps applied by the StructureLearner
    - load the causal graph that was saved by the StructureLearner
    - perform the above mentioned causal reasoning steps suggested by the *DoWhy* package

Additonally, cause2e offers helper classes for handling all paths to your data and output, representing domain knowledge, as well as bookkeeping, ranking, visualization and validation of the results of a multi-effect analysis.

## Documentation:
For a detailed documentation of the package, please refer to [mlresearchatosram.github.io/cause2e](https://mlresearchatosram.github.io/cause2e).
The documentation has been generated from Python docstrings via [Sphinx](https://www.sphinx-doc.org/en/master/).

## Outlook:
We are planning to integrate the *causal discovery toolbox* [3] as a second collection of causal discovery algorithms. In the spirit of end-to-end causal analysis, it would also be desirable to include causal representation learning before the discovery step (e.g. for image data), or causal reinforcement learning after having distilled a valid causal model that delivers interventional distributions.

## Installation:
First, install *py-causal* by following the instructions on this page: https://github.com/bd2kccd/py-causal

You can then install cause2e from pypi:
```
pip install cause2e
```
You can also install it directly from this Github repository to receive the newest version:
```
pip install dowhy -U
pip install ipython -U
pip install networkx -U
pip install numpy -U
pip install pandas -U
pip install git+git://github.com/MLResearchAtOSRAM/cause2e
```

Afterwards, try to run the [minimal example notebook](https://github.com/MLResearchAtOSRAM/cause2e/blob/main/examples/minimal_end_to_end_causal_analysis.ipynb) to check that everything works.

If you want to clone the repository into a folder for development on your local machine, please navigate to the folder and run:
```
git clone https://github.com/MLResearchAtOSRAM/cause2e
```

## Disclaimer:
cause2e is not meant to replace either *py-causal* or *DoWhy*, our goal is to make it easier for researchers to string together causal discovery and causal reasoning with these libraries. If you are only interested in causal discovery, it is preferable to directly use *py-causal* or the *TETRAD GUI*. If you are only interested in causal reasoning, it is preferable to directly use *DoWhy*.

## Citation:
If you are using cause2e in your work, please cite:

Daniel Grünbaum (2021). cause2e: A Python package for end-to-end causal analysis. https://github.com/MLResearchAtOSRAM/cause2e

## References:

[1] Chirayu (Kong) Wongchokprasitti, Harry Hochheiser, Jeremy Espino, Eamonn Maguire, Bryan Andrews, Michael Davis, & Chris Inskip. (2019, December 26). bd2kccd/py-causal v1.2.1 (Version v1.2.1). Zenodo. http://doi.org/10.5281/zenodo.3592985

[2] Amit Sharma, Emre Kiciman, et al. DoWhy: A Python package for causal inference. 2019. https://github.com/microsoft/dowhy

[3] Kalainathan, D., & Goudet, O. (2019). Causal Discovery Toolbox: Uncover causal relationships in Python. arXiv:1903.02278. https://github.com/FenTechSolutions/CausalDiscoveryToolbox


