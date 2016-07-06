# -*- coding: utf-8 -*-
import sys

from ipwxlearn.datasets.utils import split_train_valid
from ipwxlearn.glue import G
from ipwxlearn.models import ModelWithLoss, SupervisedModel, UnsupervisedModel
from ipwxlearn.models.optimizers import AdamOptimizer
from ipwxlearn.training import SummaryMonitor, ValidationMonitor, TrainingLossMonitor, run_steps


class Trainer(object):
    """
    Trainer that optimizes the model parameters.

    :param optimizer: Optimizer to train the model. (Default :class:`~ipwxlearn.model.optimizers.AdamOptimizer`).
                      See :module:`~ipwxlearn.model.optimizers` for more optimizers.
    :param batch_size: Training batch size. (Default 64)
    :param max_epoch: Maximum epoch to run for training the model. (Default 10)
    :param summary_dir: If specified, will write variable summaries to this directory. (Default None)
    :param summary_steps: Perform summary every this number of steps. (Default 100)
    :param verbose: Whether or not to print the training logs. (Default True)
    """

    def __init__(self, optimizer=AdamOptimizer(), batch_size=64, max_epoch=10, summary_dir=None, summary_steps=100,
                 verbose=True):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.summary_dir = summary_dir
        self.summary_steps = summary_steps
        self.verbose = verbose
        self.monitors = []

    def add_monitor(self, monitor):
        """Add a training-time monitor to this trainer."""
        self.monitors.append(monitor)

    def clear_monitors(self):
        """Clear all monitors."""
        self.monitors.clear()

    def fit(self, X, y=None, **kwargs):
        """
        Train the model with given data.

        :param X: Input data.
        :param y: Target data, if the model is a supervised model.

        :return: self
        """
        raise NotImplementedError()


class LossTrainer(Trainer):
    """
    Trainer that optimizes the model parameters by minimizing loss function.

    :param early_stopping: Whether or not to perform early stopping by validation? (Default True)
    :param validation_split: If a validation set is required to optimize the model, this portion
                             of data would be used as validation data. (Default 0.1)
    :param validation_steps: Perform validation every this number of steps.
                             If not specified, will automatically select the steps.
    :param validation_batch: Batch size for validation.
                             If not specified, will compute validation loss in one batch.
    :param optimizer: Optimizer to train the model. (Default :class:`~ipwxlearn.model.optimizers.AdamOptimizer`).
                      See :module:`~ipwxlearn.model.optimizers` for more optimizers.
    :param batch_size: Training batch size. (Default 64)
    :param max_epoch: Maximum epoch to run for training the model. (Default 10)
    :param verbose: Whether or not to print the training logs. (Default True)
    """

    def __init__(self, early_stopping=True, validation_split=0.1, validation_steps=None, validation_batch=None,
                 optimizer=AdamOptimizer(), batch_size=64, max_epoch=10, verbose=True):
        super(LossTrainer, self).__init__(optimizer=optimizer, batch_size=batch_size, max_epoch=max_epoch,
                                          verbose=verbose)
        self.early_stopping = early_stopping
        self.validation_split = validation_split
        self.validation_steps = validation_steps
        self.validation_batch = validation_batch

        self._loss = self._train_params = self._input_var = self._target_var = None

    def set_loss(self, loss, train_params, input_var, target_var=None):
        """
        Set the loss expression that should be minimized.

        :param loss: The loss expression.
        :param train_params: Parameters that should be trained in the loss.
        :param input_var: Input placeholder.
        :param target_var: Target placeholder, if the loss should be computed along with label.

        :return: self
        """
        self._loss = loss
        self._train_params = train_params
        self._input_var = input_var
        self._target_var = target_var
        return self

    def set_model(self, model, input_var, target_var=None, l1_reg=None, l2_reg=None, **kwargs):
        """
        Set the model that should be trained.

        :param model: Model instance.
        :param input_var: Input placeholder.
        :param target_var: Target placeholder, if the model is a supervised model.
        :param l1_reg: L1 regularization factor for this estimator. (Default None)
        :param l2_reg: L2 regularization factor for this estimator. (Default None)
        :param **kwargs: Extra arguments to be passed to :method:`get_output_for` and :method:`get_loss_for`.

        :return: self
        """
        if not isinstance(model, ModelWithLoss):
            raise TypeError('%r does not have a default loss. You should set the loss manually.')
        if isinstance(model, SupervisedModel) and target_var is None:
            raise ValueError('"target_var" should be specified for a supervised model.')
        if isinstance(model, UnsupervisedModel) and target_var is not None:
            raise ValueError('"target_var" should not be specified for an unsupervised model.')

        # derive the loss of the model, and extract trainable parameters.
        if isinstance(model, G.layers.InputLayer):
            raise TypeError('Cannot train an input layer.')
        elif isinstance(model, G.layers.MergeLayer):
            inputs = G.layers.get_output(model.input_layers, **kwargs)
        else:
            inputs = G.layers.get_output(model.input_layer, **kwargs)

        loss = model.get_loss_for(inputs, target=target_var, **kwargs)
        train_params = G.layers.get_all_params(model, trainable=True)

        # add the regularization term to the loss if required.
        if l1_reg is not None or l2_reg is not None:
            reg_params = G.layers.get_all_params(model, trainable=True, regularizable=True)
            if l1_reg is not None:
                loss += l1_reg * G.op.l1_reg(reg_params)
            if l2_reg is not None:
                loss += l2_reg * G.op.l2_reg(reg_params)

        # store the loss and parameters
        return self.set_loss(loss, train_params, input_var, target_var=target_var)

    def fit(self, X, y=None, **kwargs):
        """
        Train the model with given data.

        :param model: (model, input_var, [label_var]) or (loss, params, input_var, [label_var])

                      If a model is specified, it must be a subclass of :class:`ipwxlearn.models.ModelWithLoss`,
                      otherwise a TypeError will be thrown.
                      The third element of the tuple, `label_var`, should be specified if and only if `y` is
                      also specified.
        :param X: Input data.
        :param y: Target data, if the model is a supervised model.
        :param **kwargs: Additional arguments passed to :method:`G.layers.get_output` and
                         :method:`G.layers.get_loss_for`.

        :return: self
        """
        if self._loss is None:
            raise ValueError('You should set the loss or model before fitting on data.')
        if self._target_var is None:
            if y is not None:
                raise ValueError('An unsupervised loss function is set, but got target data %r.' % y)
            input_vars = self._input_var
            input_data = X
        else:
            if y is None:
                raise ValueError('A supervised loss function is set, but did not get target data.')
            input_vars = [self._input_var, self._target_var]
            input_data = (X, y)

        # prepare the training monitors here.
        monitors = self.monitors.copy()

        # gather summaries if required
        if self.summary_dir is not None:
            summary_writer = G.summary.SummaryWriter(self.summary_dir)
            summary = G.summary.merge_summary(G.summary.collect_variable_summaries(self._train_params))
            loss_summary = G.summary.scalar_summary('training_loss', self._loss)
            output_vars = [self._loss, loss_summary]
            monitors.append(SummaryMonitor(summary_writer, summary, steps=self.summary_steps))
        else:
            summary_writer = None
            output_vars = self._loss

        # derive update expressions for training, and compile the training function.
        updates = self.optimizer.minimize(self._loss, self._train_params)
        train_fn = G.make_function(inputs=input_vars, outputs=output_vars, updates=updates)

        # if early stopping is required, we have to build the validation function.
        # otherwise we just report the training loss.
        log_file = sys.stdout if self.verbose else None
        if self.early_stopping:
            valid_fn = G.make_function(inputs=input_vars, outputs=self._loss)
            input_data, valid_data = split_train_valid(input_data, validation_split=self.validation_split)
            monitors.append(ValidationMonitor(
                valid_fn, valid_data, params=self._train_params, steps=self.validation_steps,
                log_file=log_file, validation_batch=self.validation_batch, summary_writer=summary_writer
            ))
        else:
            monitors.append(TrainingLossMonitor(log_file=log_file, steps=self.validation_steps))

        # now it's time to run the training steps.
        max_steps = int(self.max_epoch * len(input_data) / self.batch_size)
        run_steps(G, train_fn, input_data, monitor=monitors, batch_size=self.batch_size, max_steps=max_steps)

        return self
