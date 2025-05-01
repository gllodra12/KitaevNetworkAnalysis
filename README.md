# Kitaev Network Analysis

This repository contains Python code to perform network analysis on the Kitaev chain model, as presented in _Unveiling hidden features of the Kitaev model through a complex-network analysis_ (TODO: add link)

## Features
- Compute reduced density matrices for pairs of fermions in a finite Kitaev chain.
- Construct weighted adjacency networks using quantum mutual information and concurrence measures.
- Calculate network metrics such as node density and clustering coefficient.
- Detect the topological phase transition (bulk gap closing at $|μ| = 2w$) and identify the singular fully-connected network regime.


## Installation
Create your conda environment and add:
- Python 3.8 or higher
- numpy
- scipy
- qutip
- openfermion
- networkx
- matplotlib
- rsmf

### Setup
```bash
git clone https://github.com/<username>/kitaev-network-analysis.git
cd KitaevNetworkAnalysis
```

## Usage

You can run the analysis scripts directly or explore the Jupyter notebook:

### Command-line scripts

```bash
export PYTHONPATH=$PWD
# Network analysis (Fig. 1 and Fig. 3)
./scripts/run_network_analysis.sh

# Fig A1
python ./kitaev_network_analysis/fidelity.py 14 -w 1 -d 0.5 -t 0 -pT -s 0.0
```

To generate the data for the remaining figures you must modify the command line inside `run_network_analysis.sh`.

### Jupyter Notebook
Open `paper_figures.ipynb` to load the main figures in the article.


## Citation
If you use this code in your work, please cite:
```
G. Llodrà, R. Zambrini, and G. L. Giorgi, “Unveiling hidden features of the Kitaev model through a complex-network analysis”
```

## License
This project is licensed under the MIT License.
