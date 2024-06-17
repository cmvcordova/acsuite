<div align="center">

# The Activty Cliff (AC) Suite 
[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/cmvcordova/acsuite/pulls)

AC Suite is a Pytorch Lightning + Hydra integrated utility to train and evaluate models for molecular property prediction (and beyond!) based on the Matched Molecular Pair (MMP) abstraction. Can be used to obtain embedding vectors that specifically capture the activity cliff relationship between structurally similar molecules with different activites, to enhance existing property prediction models. 

Leveraging the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template), it integrates CLI and hydra config functionality, such as multiruns, parameter sweeps, among others. Extended Connectivity Fingerprint (ECFP) based featurization is done through [molfeat's](https://github.com/datamol-io/molfeat) ```MoleculeTransformer```. Evaluation done with [MoleculeACE](https://github.com/molML/MoleculeACE).

</div>

## Installation 

### Prerequisites

- Python 3.8, 3.9, or 3.10
- [PyTorch](https://pytorch.org/get-started/locally/)
- [Hydra](https://hydra.cc/)

### Using pip

The recommended way to install the AC Suite is through ```pip```.
```
git clone https://github.com/cmvcordova/acsuite
cd acsuite
pip install .
```

## Current functionality

### Embedding pre-training

Pre-training is done as shown in the provided ```encoder_pretraining.ipynb``` notebook. Calls on the ACAModule class, from a LightningModule, to train models across a variety of tasks and objectives.

### Pre-trained encoder extraction and re-training

The ```HotSwapEncoderMLP``` optionally takes a ```pretrained_encoder_ckpt``` pointing to a model trained in the former step with an ```ACAModule```. It then extracts and freezes the pretrained encoder, placing it as the input layer of the MLP. This can then be trained with the provided ```ACAPPModule``` for both classification and regression tasks.

Examples of usage are included in the provided ```moleculeace_training-mlp/acbased.ipynb``` notebooks.

### MoleculeACE evaluation.

Currently done manually, accesses any of the 30 provided MoleculeACE ChEMBL datasets. Check out the ```ac/mlp_moleculeace_evaluation.ipynb``` notebooks that are provided for evaluation function examples. Full evaluation coming soon.

## Data

ACNet and MoleculeACE are the main data sources that are used for pre-training and downstream training and evaluation, respectively:

### ACNet
Data must be downloaded and handled by following the guidelines in the [ACNet repository](https://github.com/DrugAI/ACNet#usage). 
Generation of the datasets must be done by running `GenerateACDatasets.py` and placing the generated `JSON` files within the `ACNet` folder in the main *AC Suite* `data` directory. 

### MoleculeACE 
Data accesible through the AC Suite dataset wrapper. Installation a pre-requisite for AC Suite but can otherwise be installed following the directions over at the [MoleculeACE repo](https://github.com/datamol-io/molfeat?tab=readme-ov-file)


## Known Issues

- Hadamard, Euclidean distance Siamese classifiers not thoroughly tested, training instabilities observed. There is likely a mismatch with how they're handled within the HalfStep
- Evaluation multiruns not compatible with current script, done manually. See [Hydra issue #1258](https://github.com/facebookresearch/hydra/issues/1258).
- Pre-training data downloading method not implemented, must be dealt with manually. See ACNet section
- README could be more informative. Do not hesitate to open issues!

## Associated publication(s)
"Towards Learning Activity Cliff-Aware Molecular Representations".  
Associated thesis publication and full paper coming soon.  
Full paper/Poster accepted at [LXAI @ ICML 2024!](https://www.latinxinai.org/icml-2024)  
Poster accepted at [MoML 2024!](https://portal.ml4dd.com/moml-2024)

## How to cite

Coming soon.

## Contributing

Nothing formal here just yet, just open an issue or [drop me a line](mailto:cesar.valdezcordova@mail.mcgill.ca)

## Known Contributors

[yours truly](https://github.com/cmvcordova/)

## Acknowledgements

you for the interest and the internet

## License

This project is licensed under the MIT License - see the LICENSE file for details.
