# Hyperparameter-Free Medical Image Synthesis for Sharing Data and Improving Site-Specific Segmentation

## Setup

Create & activate a conda environment:

```
conda env create --file env_autoshare.yml
conda activate autoshare
```

Install the nnunetv2_mod:

```bash
cd nnunetv2_mod
pip install -e .
```

Set environmental variables

```
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MKL_NUM_THREADS=1
ulimit -n 500000
```

## Running experiments

```
python run_nnunet.py --config-name=cervix_000 ++data.fold=0
```

Configs for all experiments are in the config dir. Note that configs need to be modified to refer to your paths and the names/IPs of your servers (for Ray).