# -*- coding: utf-8 -*-
from __future__ import absolute_import

from ipwxlearn.utils import dataflow
from ipwxlearn.utils.misc import ensure_list_sealed, maybe_iterable_to_list
from .monitors import Monitor, MonitorChain

__all__ = [
    'run_steps'
]


def _check_monitor(monitor):
    if monitor is None:
        return Monitor()
    monitor = maybe_iterable_to_list(monitor)
    if isinstance(monitor, list):
        return MonitorChain(monitor)
    return monitor


def run_steps(train_fn, train_data, monitor=None, batch_size=32, max_steps=1000, shuffle=True):
    """
    Run determined steps to train with :param:`train_fn` and :param:`train_data`.

    :param train_fn: Callable function, which accepts one or several numpy arrays, to perform a training step.
    :param train_data: Numpy array, or a list of numpy arrays, as the training data.
    :param monitor: Monitor or a list of monitors, to guard the training process.
    :param batch_size: Mini-batch size of training.
    :param max_steps: Maximum steps to run.  If the :param:`monitor` induces early-stopping, it might
                      take less steps than this number.
    :param shuffle: Whether or not to shuffle the data after each full-pass?
    """
    # check the arguments.
    monitor = _check_monitor(monitor)
    train_data = ensure_list_sealed(train_data)
    num_examples = len(train_data[0])
    if num_examples < batch_size:
        raise ValueError('Too few data such that no training step would be performed.')

    # prepare for the training.
    monitor.start_training(batch_size, num_examples // batch_size, max_steps)

    # the out loop indicates the pass of data (or to say, the epochs)
    epoch = 0
    step = 0
    while True:
        monitor.start_epoch(epoch)
        n_batches = 0
        total_loss = 0

        # the inner loop indicates the mini-batches of data.
        for args in dataflow.iterate_training_batches(train_data, batch_size=batch_size, shuffle=shuffle):
            monitor.start_step(step)
            loss = train_fn(*args)
            monitor.end_step(step, loss)

            step += 1
            n_batches += 1
            total_loss += loss

            if step > max_steps or monitor.is_inducing_stopping:
                break

        monitor.end_epoch(epoch, float(total_loss) / n_batches)
        epoch += 1

        if step > max_steps or monitor.is_inducing_stopping:
            break

    # complete the training.
    monitor.end_training()