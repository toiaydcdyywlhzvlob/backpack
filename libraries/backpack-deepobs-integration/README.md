Integration between BackPACK and the [DeepOBS](https://github.com/abahde/DeepOBS) problems.

Requires a working installation of BackPACK, [PyTorch and TorchVision](https://pytorch.org/get-started/locally/).

For a `conda` environment installation, see `set_up_conda_env.sh`.

For a manual installation, assuming BackPACK, PyTorch and TorchVision are installed, 
```
# DeepOBS requirements
pip install argparse matplotlib2tikz numpy pandas matplotlib seaborn bayesian-optimization
pip install tikzplotlib==0.8.2
pip install palettable==3.3.0
# DeepOBS
pip install -e git+https://github.com/abahde/DeepOBS.git@2f9e658#egg=DeepOBS
pip install -e git+https://github.com/toiaydcdyywlhzvlob/backpack.git@master#egg=bp_dops_integration&subdirectory=libraries/backpack-deepobs-integration
```



