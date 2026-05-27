## Overview

SIEVE builds control-flow graph (CFG) representations for Solidity smart contracts, selects vulnerability-relevant key nodes, and trains/evaluates a graph neural network on the denoised graph. The released code is intended for reading, reproducing the main pipeline on prepared data, and inspecting the denoising case studies.

## Repository Structure

```text
.
├── keynodes_GCN_detection.py   # Detection / evaluation script
├── cfg_construct.py                     # Solidity CFG construction utilities
├── datapre.py                           # Data preprocessing utilities
├── filter_ratio.py                      # Helper script for filtering / ratio analysis
├── data/
│   ├── above0_v_count_output.csv        # Metadata/statistics of relevant elements
│   ├── cluster_simi_centers.csv         # Cluster-center metadata
│   └── datasource.txt                   # Dataset source description
├── path_to_model/
│   ├── bert-tiny/                       # Lightweight local encoder assets
│   ├── codeBert/                        # CodeBERT-related assets, if used
│   └── bert-tiny.zip
├── train_model/
│   └── key_nodes_mask_GCN.py            # GCN model definition
├── recent_ethersacn_300/
│   ├── contracts/                       # Recent contract samples
│   └── recent_contracts_manifest.csv
└── SIEVE_case_study/
    ├── *.pdf                            # Denoising visualization figures
    └── *.pt_CFGKN.pt                    # Data files of smart contract code information
```

## Environment

The code was developed with Python and PyTorch/PyTorch Geometric. Main dependencies include:

- `torch`
- `torch-geometric`
- `transformers`
- `scikit-learn`
- `pandas`
- `numpy`
- `networkx`
- `matplotlib`
- `slither-analyzer`
- `solc-select`
- `py-solc`