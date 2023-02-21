# cosmolisa
Source code for the inference of cosmological parameters with LISA observations

Download this repository, including its submodules, with

```
git clone --recursive https://github.com/wdpozzo/cosmolisa.git
```

## How to install the code with Conda

Use the environment file `environment.yml` to create the conda environment for cosmoLISA.
Run the following lines from terminal:

```
yes | conda env create -f environment.yml
conda activate cosmolisa_env
pip install ./corner
pip install ./cpnest
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
