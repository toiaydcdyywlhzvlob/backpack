"""
Special benchmarking for networks with nonlinearities

If the model contains only (piecewise-) linear functions, DiagH = DiagGGN.
Those tests are not representative of the computational complexity of DiagH.

Here, we test a network with an added Sigmoid in the last classification layer.
"""
from backpack import extensions
import pytest
import utils
from backpack.extensions.secondorder.diag_ggn.flatten import DiagGGNFlatten
from deepobs.pytorch.testproblems.testproblems_utils import flatten

"""
Load and extend all extensions to benchmark
"""

secondorder_extension_classes = {
    "KFLR": extensions.KFLR,
    "KFAC": extensions.KFAC,
    "DiagGGNExact": extensions.DiagGGNExact,
    "DiagGGNMC": extensions.DiagGGNMC,
    "DiagH": extensions.DiagHessian,
}

for name, ext_class in secondorder_extension_classes.items():
    ext_class.add_module_extension(flatten, DiagGGNFlatten())

backpack_all_extensions = {
    "Var": extensions.Variance(),
    "BatchGrad": extensions.BatchGrad(),
    "BatchL2": extensions.BatchL2Grad(),
    "SecondMoment": extensions.SumGradSquared(),
}

for name, ext_class in secondorder_extension_classes.items():
    backpack_all_extensions[name] = ext_class()

"""
Create the parameters for pytest 
"""

Ns = [2, 4, 8, 10, 16, 32, 48, 64, 96, 100, 128]

combined_parameters = []
combined_names = []
for n in Ns:
    for name, ext in backpack_all_extensions.items():
        combined_parameters.append((n, ext))
        combined_names.append(str(n) + "-" + name)

model, lossfunc, make_loader = utils.data_prep_cifar10(use_sigmoid=True)


@pytest.mark.parametrize("N", Ns)
def test_backpack_diagH(N, benchmark):
    benchmark(
        utils.make_backpack_extension_runner(
            make_loader(N), model, lossfunc, backpack_all_extensions["DiagH"]
        )
    )


@pytest.mark.parametrize("N", Ns)
def test_backpack_diagGGN(N, benchmark):
    benchmark(
        utils.make_backpack_extension_runner(
            make_loader(N), model, lossfunc, backpack_all_extensions["DiagGGNExact"]
        )
    )


@pytest.mark.parametrize("N", Ns)
def test_grad(N, benchmark):
    benchmark(
        utils.makefunc_grad(
            make_loader(N), model, lossfunc
        )
    )


"""
Use with --benchmark-min-rounds=20
"""
