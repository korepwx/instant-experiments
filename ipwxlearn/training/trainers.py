# -*- coding: utf-8 -*-
import sys

from ipwxlearn.datasets.utils import split_train_valid
from ipwxlearn.glue import G
from ipwxlearn.models import ModelWithLoss, SupervisedModel, UnsupervisedModel
from ipwxlearn.models.optimizers import AdamOptimizer
from ipwxlearn.training import SummaryMonitor, ValidationMonitor, TrainingLossMonitor, run_steps, OneShotDataFlow, \
    TestingBatchDataFlow, TrainingBatchDataFlow

__all__ = [
    'Trainer',
    'LossTrainer',
]


class Trainer(object):
    """
    Trainer that optimizes the model parameters.

    :param optimizer: Optimizer to train the model. (Default :class:`~ipwxlearn.model.optimizers.AdamOptimizer`).
                      See :module:`~ipwxlearn.model.optimizers` for more optimizers.
    :param batch_size: Training batch size. (Default 64)
    :param max_epoch: Maximum epoch to run for training the model. (Default 10)
    :param early_stopping: Whether or not to perform early stopping by validation? (Default True)
    :param validation_split: If a validation set is required to optimize the model, this portion
                             of data would be used as validation data. (Default 0.1)
    :param validation_steps: Perform validation every this number of steps.
                             If not specified, will automatically select the steps.
    :param validation_batch: Batch size for validation.
                             If not specified, will compute validation loss in one batch.
    :param summary_dir: If specified, will write variable summaries to this directory. (Default None)
    :param summary_steps: Perform summary every this number of steps. (Default 100)
    :param verbose: Whether or not to print the training logs. (Default True)
    """

    def __init__(self, optimizer=AdamOptimizer(), batch_size=64, max_epoch=10, early_stopping=True,
                 validation_split=0.1, validation_steps=None, validation_batch=None, summary_dir=None,
                 summary_steps=100, verbose=True):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.early_stopping = early_stopping
        self.validation_split = validation_split
        self.validation_steps = validation_steps
        self.validation_batch = validation_batch
        self.summary_dir = summary_dir
        self.summary_steps = summary_steps
        self.verbose = verbose
        self.monitors = []
        self._train_flow = self._valid_flow = None

    def add_monitor(self, monitor):
        """Add a training-time monitor to this trainer."""
        self.monitors.append(monitor)
        return self

    def clear_monitors(self):
        """Clear all monitors."""
        self.monitors.clear()
        return self

    def set_summary(self, summary_dir, summary_steps=100):
        """Set the training-time summary arguments."""
        self.summary_dir = summary_dir
        self.summary_steps = summary_steps
        return self

    def clear_summary(self):
        """Disable the training-time summary."""
        self.summary_dir = None
        self.summary_steps = 100
        return self

    def set_data(self, X, y=None):
        """
        Set data for this trainer.

        If early-stopping is required, the specified data will be splitted into training / validation sets,
        where the fraction of validation set is determined according to :field:`validation_split`.
        The training data will be shuffled before each epoch, so if this does not satisfy your demands,
        you may set custom data flow objects by :method:`set_data_flow`.

        :param X: Input data.
        :param y: Target data, if the model is a supervised model.
        :return: self
        """
        input_data = X if y is None else (X, y)
        if self.early_stopping:
            # If early stopping is required, we should construct the validation data flow.
            input_data, valid_data = split_train_valid(input_data, validation_split=self.validation_split)
            if self.validation_batch is None:
                valid_flow = OneShotDataFlow(valid_data)
            else:
                valid_flow = TestingBatchDataFlow(valid_data, batch_size=self.validation_batch)
        else:
            valid_flow = None
        train_flow = TrainingBatchDataFlow(input_data, batch_size=self.batch_size)
        return self.set_data_flow(train_flow, valid_flow)

    def set_data_flow(self, train_flow, valid_flow=None):
        """
        Set data flow for this trainer.

        :param train_flow: Training data flow.
        :param valid_flow: Validation data flow. Must be specified if early-stopping is required.
        :return: self
        """
        if valid_flow is None and self.early_stopping:
            raise ValueError('Early stopping requires validation data flow.')
        self._train_flow = train_flow
        self._valid_flow = valid_flow
        return self

    def set_model(self, model, input_var, target_var=None, l1_reg=None, l2_reg=None, **kwargs):
        """
        Set the model that should be trained.

        :param model: Model instance.
        :param input_var: Input placeholder.
        :param target_var: Target placeholder, if the model is a supervised model.
        :param l1_reg: L1 regularization factor for the parameters of this estimator. (Default None)
        :param l2_reg: L2 regularization factor for the parameters of this estimator. (Default None)
        :param **kwargs: Extra arguments to be passed to :method:`get_output_for` and :method:`get_loss_for`.

        :return: self
        """
        raise NotImplementedError()

    def fit(self, X=None, y=None):
        """
        Train the model with given data.

        If :param:`X` and :param:`y` are specified, will call :method:`set_data` to override
        the training data.  Otherwise will use the training data already set before.

        :param X: Input data.
        :param y: Target data, if the model is a supervised model.

        :return: self
        """
        raise NotImplementedError()


class LossTrainer(Trainer):
    """
    Trainer that optimizes the model parameters by minimizing loss function.
    See :class:`Trainer` for details of arguments.
    """

    def __init__(self, *args, **kwargs):
        super(LossTrainer, self).__init__(*args, **kwargs)

        self._loss = self._train_params = self._input_var = self._target_var = \
            self._input_vars = self._train_fn = self._summary = None

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

        # check the arguments and prepare for training function
        if target_var is None:
            self._input_vars = self._input_var
        else:
            self._input_vars = [self._input_var, self._target_var]

        # gather summaries
        loss_summary = G.summary.scalar_summary('training_loss', self._loss)
        self._summary = G.summary.merge_summary(G.summary.collect_variable_summaries(self._train_params))
        output_vars = [self._loss, loss_summary]

        # derive update expressions for training, and compile the training function.
        updates = self.optimizer.minimize(self._loss, self._train_params)
        self._train_fn = G.make_function(inputs=self._input_vars, outputs=output_vars, updates=updates)

        return self

    def set_model(self, model, input_var, target_var=None, l1_reg=None, l2_reg=None, **kwargs):
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

        loss = G.op.mean(model.get_loss_for(inputs, target=target_var, **kwargs))
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

    def set_data_flow(self, train_flow, valid_flow=None):
        if valid_flow is not None:
            if train_flow.array_count != valid_flow.array_count:
                raise ValueError('Number of arrays returned by training and validation data flow at each mini-batch '
                                 'does not agree.')
        return super(LossTrainer, self).set_data_flow(train_flow=train_flow, valid_flow=valid_flow)

    def fit(self, X=None, y=None):
        if self._loss is None:
            raise ValueError('You should set the loss or model before fitting on data.')
        if self._target_var is None:
            if y is not None:
                raise ValueError('An unsupervised loss function is set, but got target data %r.' % y)
        else:
            if y is None:
                raise ValueError('A supervised loss function is set, but did not get target data.')
        if X is None and y is not None:
            raise ValueError('Specifying target data without input data is meaningless.')

        # override the training data.
        if X is not None:
            self.set_data(X, y)

        # validate whether or not we've got right number of arrays from the data flow.
        if self._train_flow.array_count == 1:
            if self._target_var is not None:
                raise ValueError('Supervised model requires target data for training.')
        elif self._train_flow.array_count == 2:
            if self._target_var is None:
                raise ValueError('Unsupervised model does not need target data for training.')
        else:
            raise ValueError('Too many arrays returned by training and validation data flow at each mini-batch.')

        # prepare the training monitors here.
        monitors = self.monitors.copy()

        # add summary monitor if required
        if self.summary_dir is None:
            summary_writer = None
        else:
            summary_writer = G.summary.SummaryWriter(self.summary_dir)
            monitors.append(SummaryMonitor(summary_writer, self._summary, steps=self.summary_steps))

        # if validation set is specified, we have to build the validation function.
        # otherwise we just report the training loss.
        log_file = sys.stdout if self.verbose else None
        if self._valid_flow is not None:
            valid_fn = G.make_function(inputs=self._input_vars, outputs=self._loss)
            monitors.append(ValidationMonitor(
                valid_fn, self._valid_flow, params=self._train_params, steps=self.validation_steps,
                log_file=log_file, validation_batch=self.validation_batch, summary_writer=summary_writer
            ))
        else:
            monitors.append(TrainingLossMonitor(log_file=log_file, steps=self.validation_steps))

        # now it's time to run the training steps.
        max_steps = int(self.max_epoch * len(X) / self.batch_size)
        run_steps(G, self._train_fn, self._train_flow, monitor=monitors, batch_size=self.batch_size,
                  max_steps=max_steps, summary_writer=summary_writer)

        return self
