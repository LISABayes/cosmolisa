# cosmolisa
Source code for the inference of cosmological parameters with LISA observations

## How to install the code with Conda

Create an environment file `environment.yml` with the following lines:

```
name: my_fancy_env_name
channels:
- default
- conda-forge
dependencies:
  - python=3.9
  - matplotlib=3.5.1
  - numpy=1.22.0
  - cython=0.29.26
  - tqdm=4.62.3
  - scipy=1.7.3
  - corner=2.2.1
  - setuptools=60.5.0
  - psutil=5.9.0
  - lalsuite=7.2
```

Run the following lines from terminal:

```
yes | conda env create -f environment.yml
conda activate my_fancy_env_name
pip install git+https://github.com/johnveitch/cpnest@massively_parallel
export LAL_PREFIX=/place/of/your/home/directory/.conda/envs/my_fancy_env_name
python setup.py install
```

If the installation ends without errors, the code is ready for use. 

You should be able to visualise the help message by giving the following one-line command:

```
cosmoLISA --help
```

To run the EMRI example (or any other analysis), specify the config file with the `--config-file` option:

```
cosmoLISA --config-file /path/where/you/saved/the/repository/cosmolisa/cosmolisa/config_EMRI.ini
```
