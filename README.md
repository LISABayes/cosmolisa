# cosmolisa
Source code for the inference of cosmological parameters with LISA observations

Download this repository, with

```
git clone https://github.com/LISABayes/cosmolisa.git
```

## How to install the code with Conda

Use the environment file `environment.yml` to create the conda environment for cosmoLISA.
Run the following lines from terminal:

```
yes | conda env create -f environment.yml
conda activate cosmolisa_env
pip install raynest==1.0.1
export LAL_PREFIX=$HOME/.conda/envs/cosmolisa_env
python setup.py install
```

If the installation ends without errors, the code is ready for use. 

You should be able to visualise the help message by giving the following one-line command:

```
cosmoLISA --help
```

To run any analysis, specify the (relative or absolute) path of the config file with the `--config-file` option.
To run the EMRI example:

```
cosmoLISA --config-file config_EMRI.ini
```

Authors: Walter Del Pozzo, Danny Laghi
