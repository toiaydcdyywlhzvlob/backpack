# Reproduce our experiments

## Roadmap
1. Create the run scripts for grid search
2. Run the grid search (**alternatively** extract our data)
3. (Optional) verify completeness of the grid search
4. Create the run scripts for repetition of the best run
5. Rerun the best setting for 10 different random seeds (**alternatively** extract our data)
6. (Optional) verify completeness of best runs
7. Plotting

## Installation Preliminaries

First, install [PyTorch and TorchVision](https://pytorch.org/get-started/locally/) if you do not have them. 

You also need additional dependencies to use `DeepOBS` (copied from its website):
```bash
pip install argparse matplotlib2tikz numpy pandas matplotlib seaborn bayesian-optimization
```

Then, we'll be able to install BackPACK and the libraries required for the experiments:

* Clone our repository and change into the `libraries/` directory
* Install the dependencies (`BackPACK`, `BackPACK-Optim`, `DeepOBS`, and the integration package) 
```bash
pip install -e backpack/
pip install -e backpack-optim/
pip install -e backpack-deepobs-integration/
pip install -e git+https://github.com/abahde/DeepOBS.git@2f9e658#egg=DeepOBS
pip install tikzplotlib==0.8.2
pip install palettable==3.3.0
```

> **Note:** From this point, all shell commands are assumed to be executed in `experiments/exp`.


## Grid search
> **Note**: You can skip this step by extracting `grid_search.zip` in `experiments/grid_search`
> ```bash
> unzip ../grid_search.zip -d ../
> ```

### Run scripts

The `python` jobs for the grid search are written into `.txt` files in `experiments/grid_search_command_scripts`. To create them, run
```bash
python exp01_grid_search.py
```
You can restrict to a single `DeepOBS` testproblem by specifying the `--dobs_problem` option. If not specified, all run scripts will be created:
```bash
python exp01_grid_search.py --help

usage: exp01_grid_search.py [-h]
                            [--dobs_problem {mnist_logreg,fmnist_2c2d,cifar10_3c3d,cifar100_allcnnc,all}]

Choose the `DeepOBS` problem

optional arguments:
  -h, --help            show this help message and exit
  --dobs_problem {mnist_logreg,fmnist_2c2d,cifar10_3c3d,cifar100_allcnnc,all}
```

**Example**: Create the run scripts for `mnist_logreg`:
```bash
python exp01_grid_search.py --dobs_problem mnist_logreg
```
This results in the following files:
```
ls ../grid_search_command_scripts

...
jobs_DiagGGNConstantDampingOptimizer_grid_search_mnist_logreg.txt
jobs_DiagGGNMCConstantDampingOptimizer_grid_search_mnist_logreg.txt
jobs_KFACConstantDampingOptimizer_grid_search_mnist_logreg.txt
jobs_KFLRConstantDampingOptimizer_grid_search_mnist_logreg.txt
jobs_KFRAConstantDampingOptimizer_grid_search_mnist_logreg.txt
...
```

> **Note:** You can remove previously created run scripts by executing `clean_run_scripts.sh`

### Execution

To execute runs of the grid search, navigate to `code/grid_search_command_scripts` and run all `python` jobs in the `.txt` files. For example, running the jobs for `DiagGGN` on `mnist_logreg`:
```bash
bash jobs_DiagGGNConstantDampingOptimizer_grid_search_mnist_logreg.txt
```

> **Warning**: Running all jobs sequentially leads to **very long run times**, even on `mnist_logreg`. Consider using our data instead.

## (Optional) Verify completeness of grid search
Run
```bash
python exp02_check_grid_search
```
which will provide a summary table of the grid search status:
```
------------------------------------------------------------------------------
GRID SEARCH TEST SUMMARY

          Problem  Found all  Grid complete
     mnist_logreg       True           True
      fmnist_2c2d       True           True
     cifar10_3c3d       True           True
 cifar100_allcnnc       True          False

Found runs evaluated on single seed: True
------------------------------------------------------------------------------
```
In the example above, all runs except `cifar100_allcnnc` are complete.

## Rerun best hyperparameter setting
> **Note**: You can skip this step by extracting `best_run.zip` in `experiments/best_run`
> ```bash
> unzip ../best_run.zip -d ../
> ```

The `DeepOBS` benchmarking protocol suggests to repeat the best hyperparameter setting for different random seeds. The training metrics shown in the final results correspond to the statistics over these realizations.

### Run scripts
Creating the run scripts can be done in complete analogy to the grid search. For `mnist_logreg` only:
```
python exp03_best_run.py --dobs_problem mnist_logreg
```
The jobs are generated in `code/best_run_command_scripts`
```
ls ../best_run_command_scripts

...
jobs_mnist_logreg_DiagGGN_const_final_valid_accuracies.txt
jobs_mnist_logreg_SDiagGGN_const_final_valid_accuracies.txt
jobs_mnist_logreg_KFAC_const_final_valid_accuracies.txt
jobs_mnist_logreg_KFLR_const_final_valid_accuracies.txt
jobs_mnist_logreg_KFRA_const_final_valid_accuracies.txt
...
```

### Execution
Navigate to `experiments/best_run_command_scripts` and run all `python` jobs in the `.txt` files. For example, running the jobs for `DiagGGN` on `mnist_logreg`:
```bash
bash jobs_mnist_logreg_DiagGGN_const_final_valid_accuracies.txt
```

> **Warning**: Running all jobs sequentially leads to **very long run times**, even on `mnist_logreg`. Consider using our data instead. Use

## (Optional) Verify completeness of best runs
The command
```bash
python exp04_check_best_run.py
```
checks if all optimizers have been evaluated on 10 different random seeds. Example output for missing runs on `cifar100_allcnnc`:

```
------------------------------------------------------------------------------
BEST RUN TEST SUMMARY

          Problem Found all optims
     mnist_logreg           [True]
      fmnist_2c2d           [True]
     cifar10_3c3d           [True]
 cifar100_allcnnc          [False]

Found runs evaluated on 10 different seeds: [False]
------------------------------------------------------------------------------
```

## Plotting
We use the `DeepOBS` functionality to plot the results and save (low quality) `.png` previews of the metrics during training.

Make sure you have the `DeepOBS` baselines `baselines.zip` extracted to `experiments/baselines`.

> ```bash
> unzip ../baselines.zip -d ../
> ```

Plotting the results of `mnist_logreg`:
```bash
python exp05_plot_different_curvatures.py --dobs_problem mnist_logreg
```
The plots can be found in `experiments/plots/performance/problem/curvatures_const_final_valid_accuracies`. For the presentation in the paper, we modified the `.tex` *TikZ* plots to our needs.
