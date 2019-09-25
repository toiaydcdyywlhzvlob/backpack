"""Handle tunable parameters of optimizer or damping."""


class TuningBase():
    """Base class for tuning hyperparameters.

    Parameters:
    -----------
    hyperparams: dict
        Nested dictionary that lists the tunable hyperparameters and data
        types, e.g. {"lr": {"type": float}, ...}.
    grid : dict
        Nested dictionary mapping tunable hyperparameter names to values, e.g.
        {"lr": [0.001, 0.01, 0.1], ...}. The grid is defined by the Cartesian
        product of values
    """
    def __init__(self, hyperparams=None, grid=None):
        if hyperparams is None:
            hyperparams = self.default_hyperparams()
        self.hyperparams = hyperparams

        if grid is None:
            grid = self.default_grid()
        self.grid = grid

    def default_hyperparams(self):
        raise NotImplementedError

    def default_grid(self):
        raise NotImplementedError

    def get_hyperparams(self):
        self._verify_hyperparams()
        return self.hyperparams

    def get_grid(self):
        self._verify_grid()
        return self.grid

    def _verify_hyperparams(self):
        """Do not allow default values."""
        DEFAULT = "default"
        has_default = []
        for param, param_prop in self.hyperparams.items():
            if DEFAULT in param_prop.keys():
                has_default.append(param)

        throw_exception = len(has_default) != 0
        if throw_exception:
            raise ValueError(
                "Parameters {} have default value.".format(has_default))

    def _verify_grid(self):
        """Grid has to be specified for all parameters."""
        not_specified = []
        for param in self.hyperparams.keys():
            if param not in self.grid.keys():
                not_specified.append(param)

        throw_exception = len(not_specified) != 0
        if throw_exception:
            raise ValueError(
                "Parameters {} not specified in grid.".format(not_specified))
