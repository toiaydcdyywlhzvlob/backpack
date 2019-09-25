"""Runners for DeepOBS integration with BackPACK optimizers.

Notes:
------
- BackPACK optimizers require a closure
- The loss function has to correspond to an empirical risk
  minimization objective. Therefore, regularization has to be handled
  internally by the optimizer
- L2 regularization is supported. The optimizer has to be initialized
  with a `param_groups` dictionary
"""

import copy
import warnings

import numpy as np
import torch

from backpack import extend
from deepobs.pytorch.runners import PTRunner


class BatchLossIsNanOrTooLargeException(Exception):
    """Raised if non-finite value is encountered in batch loss."""
    def __init__(self, batch_loss):
        self.batch_loss = batch_loss
        message = "Batch loss is NaN or too large: {}".format(batch_loss)
        super().__init__(message)


class MoreThanOneBatchUsedException(Exception):
    """Raised if non-finite value is encountered in batch loss."""
    def __init__(self, before, after):
        self.before = before
        self.after = after
        message = "Used {} batches (before: {}, after {}), but only one allowed".format(
            after - before, before, after)
        super().__init__(message)


class BaseRunner(PTRunner):
    """Standard runner, extract steps for BackPACK integration.

    Three functions need to be implemented:
    1) `build_optimizer`: Initializes the optimizer with the
        parameter group dictionary before training starts.
    2) `_perform_step`: Perform a step of the optimizer, return
       the batch loss
    3) `_preprocess`: Preprocess the DeepOBS problem (optional)
    """
    def _build_optimizer(self, hyperparams, tproblem):
        params = tproblem.net.parameters()
        optimizer = self._optimizer_class(params, **hyperparams)
        return optimizer

    def _print_optimizer_param_groups(self, optimizer):
        """For verification of parameter assignment."""
        def tabs(num, prefix="[DEBUG] "):
            return prefix + num * "\t"

        print("{}Optimizer parameter groups".format(tabs(0)))
        for idx, group in enumerate(optimizer.param_groups):
            print('{}Group {}:'.format(tabs(0), idx))
            for key, value in group.items():
                print_value = value
                if key is "params":
                    print_value = "".join([
                        "\n{}{}".format(tabs(1), tuple(p.size()))
                        for p in value
                    ])
                print("{}{}: {}".format(tabs(1), key, print_value))
        print("{}End of optimizer parameter groups".format(tabs(0)))

    def _perform_step(self, optimizer, tproblem):
        """Return batch loss and dictionary with information about the step.

        The information has to be returned as a dictionary with lists as values.
        """
        optimizer.zero_grad()
        batch_loss, _ = tproblem.get_batch_loss_and_accuracy()

        if self._is_nan_or_too_large(batch_loss):
            raise BatchLossIsNanOrTooLargeException(batch_loss)

        batch_loss.backward()
        optimizer.step()
        step_info = {}

        return batch_loss, step_info

    def _is_nan_or_too_large(self, loss, limit=1e10):
        is_nan = not np.isfinite(loss.item())
        too_large = loss.item() > limit
        return is_nan or too_large

    def _preprocess(self, tproblem, backpack_debug):
        return tproblem

    def training(self,
                 tproblem,
                 hyperparams,
                 num_epochs,
                 print_train_iter,
                 train_log_interval,
                 tb_log=None,
                 tb_log_dir=None,
                 backpack_debug=False,
                 **train_params):
        """This is a copy of the DeepOBS StandardRunner training.

        Entry points for the integration with BackPACK have been extracted
        into methods.
        """
        # Construction of optimizer and optimization step are extracted
        tproblem = self._preprocess(tproblem, backpack_debug)

        opt = self._build_optimizer(hyperparams, tproblem)
        self._print_optimizer_param_groups(opt)

        # Lists to log train/test loss and accuracy.
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []

        minibatch_train_losses = []

        if tb_log:
            try:
                from torch.utils.tensorboard import SummaryWriter
                summary_writer = SummaryWriter(log_dir=tb_log_dir)
            except ImportError as e:
                warnings.warn(
                    'Not possible to use tensorboard for pytorch. Reason: ' +
                    e.msg, RuntimeWarning)
                tb_log = False
        global_step = 0

        custom_log = {}
        for epoch_count in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            self.evaluate_all(
                epoch_count,
                num_epochs,
                tproblem,
                train_losses,
                valid_losses,
                test_losses,
                train_accuracies,
                valid_accuracies,
                test_accuracies,
            )

            # Break from train loop after the last round of evaluation
            if epoch_count == num_epochs:
                break

            ### Training ###

            # set to training mode
            tproblem.train_init_op()
            batch_count = 0

            while True:
                try:
                    batches_before = tproblem._batch_count
                    batch_loss, step_info = self._perform_step(opt, tproblem)
                    batches_after = tproblem._batch_count

                    if batches_after - batches_before != 1:
                        raise MoreThanOneBatchUsedException(
                            batches_before, batches_after)

                    if batch_count % train_log_interval == 0:
                        minibatch_train_losses.append(batch_loss.item())
                        if print_train_iter:
                            print("Epoch {0:d}, step {1:d}: loss {2:g}".format(
                                epoch_count, batch_count, batch_loss))
                        if tb_log:
                            summary_writer.add_scalar('loss',
                                                      batch_loss.item(),
                                                      global_step)

                    batch_count += 1
                    global_step += 1

                except BatchLossIsNanOrTooLargeException as e:
                    print(e)
                    batch_loss = np.array([float('nan')])

                    if len(minibatch_train_losses) == 0:
                        minibatch_train_losses.append(e.batch_loss.item())

                    self._update_custom_log(step_info, custom_log)
                    self._print_step_info(step_info)
                    break
                except StopIteration:
                    self._update_custom_log(step_info, custom_log)
                    self._print_step_info(step_info)
                    break

            if not np.isfinite(batch_loss.item()):
                self._abort_routine(
                    epoch_count,
                    num_epochs,
                    train_losses,
                    valid_losses,
                    test_losses,
                    train_accuracies,
                    valid_accuracies,
                    test_accuracies,
                    minibatch_train_losses,
                )
                break
            else:
                continue

        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            'valid_losses': valid_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            'valid_accuracies': valid_accuracies,
            "test_accuracies": test_accuracies,
            **custom_log,
        }

        return output

    @staticmethod
    def _print_step_info(info):
        prefix = "[INFO]"

        def entry(name, value):
            return '{}\t{:<35}: {}'.format(prefix, name, value)

        if info:
            print("{} Extended metrics of last step:".format(prefix))
            for key, value in info.items():
                print(entry(key, value))

    def _update_custom_log(self, step_info, log):
        """Append the quantities of `step_info` to `log`."""
        def update(key):
            value = step_info[key]
            key_exists = key in log.keys()

            if key_exists:
                log[key].append(value)
            else:
                log[key] = [value]

        for key in step_info.keys():
            update(key)


class NoBackwardOnRegularizationRunner(BaseRunner):
    """Do not call backward of L2 regularization.

    Handle L2 regularization internally with the optimizer.
    """
    def _get_param_groups_with_weight_decay(self, tproblem):
        param_groups_with_wd = []
        for weight_decay, params in tproblem.regularization_groups.items():
            param_group = {
                "params": params,
                "weight_decay": weight_decay,
            }
            param_groups_with_wd.append(param_group)

        return param_groups_with_wd

    def _build_optimizer(self, hyperparams, tproblem):
        """Extract weight_decay from problem and inform optimizer."""
        param_groups = self._get_param_groups_with_weight_decay(tproblem)
        return self._optimizer_class(param_groups, **hyperparams)

    def _perform_step(self, optimizer, tproblem):
        """Evaluate gradients only on empirical risk."""
        reg_loss = self._get_l2_reg_loss(tproblem)
        closure = self._create_closure(tproblem, optimizer)

        batch_loss = optimizer.step(closure)
        step_info = {}

        return batch_loss + reg_loss, step_info

    def _create_closure(self, tproblem, optimizer):
        """Evaluate only empirical risk, not the regularization loss"""
        def closure():
            optimizer.zero_grad()
            batch_loss, _ = tproblem.get_batch_loss_and_accuracy(
                add_regularization_if_available=False)

            if self._is_nan_or_too_large(batch_loss):
                raise BatchLossIsNanOrTooLargeException(batch_loss)

            batch_loss.backward()
            return batch_loss

        return closure

    def _get_l2_reg_loss(self, tproblem):
        with torch.no_grad():
            reg_loss = tproblem.get_regularization_loss()
        return reg_loss


class BPOptimRunner(NoBackwardOnRegularizationRunner):
    """Do not call backward of regularization.

    Do not repeat runs that have already been executed.
    """
    def _create_closure(self, tproblem, optimizer):
        """Return loss and model output (required for HVPs)."""
        forward_func = tproblem.get_batch_loss_and_accuracy_func(
            add_regularization_if_available=False)

        def closure():
            """Forward pass of the empirical risk.

            In contrast to the PyTorch docs, the closure required for the
            optimizers in BPOptim should

            * not clear the gradients
            * not call backward() on the loss
            """
            batch_loss, batch_accuracy = forward_func()

            if self._is_nan_or_too_large(batch_loss):
                raise BatchLossIsNanOrTooLargeException(batch_loss)

            # model output with BackPACK
            model_output = tproblem.net.output

            return batch_loss, model_output

        return closure

    def _preprocess(self, tproblem, backpack_debug):
        """Make model and loss function BackPACKable."""
        extend(tproblem.net, debug=backpack_debug)
        tproblem._old_loss = tproblem.loss_function

        def hotfix_lossfunc(reduction="mean"):
            return extend(tproblem._old_loss(reduction=reduction),
                          debug=backpack_debug)

        tproblem.loss_function = hotfix_lossfunc
        return tproblem

    def run(self,
            testproblem=None,
            hyperparams=None,
            batch_size=None,
            num_epochs=None,
            random_seed=None,
            data_dir=None,
            output_dir=None,
            weight_decay=None,
            no_logs=None,
            train_log_interval=None,
            print_train_iter=None,
            tb_log=None,
            tb_log_dir=None,
            skip_if_exists=True,
            **training_params):
        """Copy from DeepOBS, Do not repeat identical runs.

        Need to change keyword argument `skip_if_exist to `True`.
        """
        return super().run(testproblem=testproblem,
                           hyperparams=hyperparams,
                           batch_size=batch_size,
                           num_epochs=num_epochs,
                           random_seed=random_seed,
                           data_dir=data_dir,
                           output_dir=output_dir,
                           weight_decay=weight_decay,
                           no_logs=no_logs,
                           train_log_interval=train_log_interval,
                           print_train_iter=print_train_iter,
                           tb_log=tb_log,
                           tb_log_dir=tb_log_dir,
                           skip_if_exists=skip_if_exists,
                           **training_params)


class BPOptimRunnerExtendedLogging(BPOptimRunner):
    """Logging of additional quantities during training."""

    BATCH_LOSS_BEFORE = "batch_loss_before_step"
    BATCH_LOSS_AFTER = "batch_loss_after_step"
    L2_REG_BEFORE = "l2_reg_before_step"
    L2_REG_AFTER = "l2_reg_after_step"
    BATCH_LOSS_GRAD_NORM_BEFORE = "batch_loss_grad_norm_before_step"
    BATCH_LOSS_GRAD_NORM_AFTER = "batch_loss_grad_norm_after_step"
    PARAM_CHANGE_NORM = "parameter_change_norm"
    BATCH_LOSS_IMPROVEMENT = "batch_loss_improvement"

    COMPUTE_BEFORE = [
        BATCH_LOSS_BEFORE,
        L2_REG_BEFORE,
        BATCH_LOSS_GRAD_NORM_BEFORE,
    ]

    COMPUTE_AFTER = [
        BATCH_LOSS_AFTER,
        L2_REG_AFTER,
        BATCH_LOSS_GRAD_NORM_AFTER,
        PARAM_CHANGE_NORM,
    ]

    CONSTRUCT_AFTER_COMPUTATION = [
        BATCH_LOSS_IMPROVEMENT,
    ]

    def _perform_step(self, optimizer, tproblem):
        """Evaluate gradients only on empirical risk."""
        reg_loss = self._get_l2_reg_loss(tproblem)
        closure = self._create_closure(tproblem, optimizer)

        params_before = copy.deepcopy(list(tproblem.net.parameters()))
        before_info = self._compute_metrics_before_step(
            optimizer, closure, tproblem)

        batch_loss = optimizer.step(closure)
        step_info = optimizer.get_step_info()

        params_after = list(tproblem.net.parameters())
        after_info = self._compute_metrics_after_step(optimizer, closure,
                                                      tproblem, params_before,
                                                      params_after)
        reconstruct_info = self._reconstruct_metrics(before_info, after_info)

        full_step_info = self._combine_step_info(before_info, step_info,
                                                 after_info, reconstruct_info)

        return batch_loss + reg_loss, full_step_info

    def _compute_metrics_before_step(self, optimizer, closure_before_step,
                                     tproblem):
        metrics = {}
        for metric in self.COMPUTE_BEFORE:
            if metric == self.BATCH_LOSS_BEFORE:
                with torch.no_grad():
                    value = self.__compute_metric_batch_loss(
                        closure_before_step).item()
            elif metric == self.L2_REG_BEFORE:
                value = self._get_l2_reg_loss(tproblem)
            elif metric == self.BATCH_LOSS_GRAD_NORM_BEFORE:
                value = self.__compute_metric_grad_norm(
                    optimizer, tproblem, closure_before_step).detach().item()
            else:
                raise NotImplementedError
            metrics[metric] = value

        return metrics

    def _compute_metrics_after_step(self, optimizer, closure_after_step,
                                    tproblem, params_before, params_after):
        metrics = {}
        for metric in self.COMPUTE_AFTER:
            if metric == self.BATCH_LOSS_AFTER:
                with torch.no_grad():
                    value = self.__compute_metric_batch_loss(
                        closure_after_step).item()
            elif metric == self.L2_REG_AFTER:
                value = self._get_l2_reg_loss(tproblem)
            elif metric == self.BATCH_LOSS_GRAD_NORM_AFTER:
                value = self.__compute_metric_grad_norm(
                    optimizer, tproblem, closure_after_step).detach().item()
            elif metric == self.PARAM_CHANGE_NORM:
                value = self.__compute_metric_param_change_norm(
                    params_after, params_before).detach().item()
            else:
                raise NotImplementedError
            metrics[metric] = value

        return metrics

    def _reconstruct_metrics(self, before_metrics, after_metrics):
        reconstructed = {}
        for metric in self.CONSTRUCT_AFTER_COMPUTATION:
            if metric == self.BATCH_LOSS_IMPROVEMENT:
                value = after_metrics[self.BATCH_LOSS_AFTER] - before_metrics[
                    self.BATCH_LOSS_BEFORE]
            reconstructed[metric] = value

        return reconstructed

    def __compute_metric_param_change_norm(self, params_after, params_before):
        param_change = self.__get_param_change(params_after, params_before)
        return self.__vectorize_and_compute_l2norm(param_change)

    @staticmethod
    def __get_param_change(new_params, old_params, detach=True):
        """Return parameter differences p_new - p_old."""
        param_change = []
        for p_new, p_old in zip(new_params, old_params):
            dp = p_new - p_old
            if detach:
                dp = dp.detach()
            param_change.append(dp)
        return param_change

    def __compute_metric_grad_norm(self, optimizer, tproblem, closure):
        optimizer.zero_grad()
        loss = self.__compute_metric_batch_loss(closure)
        loss.backward()
        grads = [p.grad for p in tproblem.net.parameters()]
        return self.__vectorize_and_compute_l2norm(grads)

    @staticmethod
    def __compute_metric_batch_loss(closure):
        batch_loss, _ = closure()
        return batch_loss

    @staticmethod
    def __vectorize_and_compute_l2norm(list_of_tensors):
        vec = torch.cat([t.flatten() for t in list_of_tensors])
        return torch.norm(vec)

    @staticmethod
    def _combine_step_info(before, step, after, reconstructed):
        not_computed_during_step = {
            **before,
            **after,
            **reconstructed,
        }
        extra_info_from_step = {}

        for key, value in step.items():
            if key in not_computed_during_step.keys():
                # verify same value
                comparison_value = not_computed_during_step[key]
                if not value == comparison_value:
                    raise ValueError("Mismatch in {}: {} != {}".format(
                        key, value, comparison_value))
            else:
                extra_info_from_step[key] = value

        return {
            **not_computed_during_step,
            **extra_info_from_step,
        }
