import deepobs
from backpack import extensions
import pytest
import utils
from backpack.extensions.secondorder.diag_ggn.flatten import DiagGGNFlatten
from deepobs.pytorch.testproblems.testproblems_utils import flatten

model, lossfunc, make_loader = utils.data_prep_cifar10()

backpack_secondorder_extension_classes = {
    #    "KFRA": extensions.KFRA,
    "KFLR": extensions.KFLR,
    "KFAC": extensions.KFAC,
    "DiagGGNExact": extensions.DiagGGNExact,
    "DiagGGNMC": extensions.DiagGGNMC,
    "DiagH": extensions.DiagHessian,
}

backpack_extensions = {
    "Var": extensions.Variance(),
    "BatchGrad": extensions.BatchGrad(),
    "BatchL2": extensions.BatchL2Grad(),
    "SecondMoment": extensions.SumGradSquared(),
}

for name, ext_class in backpack_secondorder_extension_classes.items():
    ext_class.add_module_extension(flatten, DiagGGNFlatten())
    backpack_extensions[name] = ext_class()

combined_parameters = []
combined_names = []
for n in utils.Ns:
    for name, ext in backpack_extensions.items():
        combined_parameters.append((n, ext))
        combined_names.append(str(n) + "-" + name)


@pytest.mark.parametrize("N, ext", combined_parameters, ids=combined_names)
def test_cifar10_backpack_extension(N, ext, benchmark):
    benchmark(
        utils.make_backpack_extension_runner(
            make_loader(N), model, lossfunc, ext
        )
    )


@pytest.mark.parametrize("N", utils.Ns)
def test_cifar10_batchgrad_forloop(N, benchmark):
    benchmark(
        utils.makefunc_batchgrad_forloop(
            make_loader(N), model, lossfunc
        )
    )


@pytest.mark.parametrize("N", utils.Ns)
def test_cifar10_grad(N, benchmark):
    benchmark(
        utils.makefunc_grad(
            make_loader(N), model, lossfunc
        )
    )


"""
@pytest.mark.parametrize("N", utils.Ns)
def test_cifar10_backpack_diagH_sigmoid(N, benchmark):
    model, lossfunc, make_loader = utils.data_prep_fmnist_sigmoid()
    benchmark(
        utils.make_backpack_extension_runner(
            make_loader(N), model, lossfunc, backpack_extensions["DiagGGNExact"]
        )
    )
"""