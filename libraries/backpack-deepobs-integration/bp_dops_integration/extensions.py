"""Inform BackPACK extensions how to deal with DeepOBS-specific layers."""
from backpack.extensions import DiagGGN
from backpack.extensions.secondorder.diag_ggn.flatten import DiagGGNFlatten
from deepobs.pytorch.testproblems.testproblems_utils import flatten


def extend_deepobs_flatten():
    """Inform BackPACK how to deal with DeepOBS flatten layer."""
    print("[DEBUG] BackPACK: Extend DeepOBS layer flatten")
    DiagGGN.add_module_extension(flatten, DiagGGNFlatten())
