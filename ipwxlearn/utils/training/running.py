# -*- coding: utf-8 -*-
from __future__ import absolute_import

from ipwxlearn.utils.dataflow import DataFlow, TrainingBatchDataFlow
from ipwxlearn.utils.misc import maybe_iterable_to_list
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


def run_steps(G, train_fn, train_data, monitor=None, batch_size=32, max_steps=1000, shuffle=True, summary_writer=None):
    """
    Run determined steps to train with :param:`train_fn` and :param:`train_data`.

    :param G: The tensor backend.
    :param train_fn: Callable function, which accepts one or several numpy arrays, to perform a training step.
                     This function should either return a scalar which indicates the training loss,
                     or return a tuple which contains not only the training loss, but also the summary object
                     for the loss.
    :param train_data: Numpy array, a list of numpy arrays, or a DataFlow object as the training data.
    :param monitor: Monitor or a list of monitors, to guard the training process.
    :param batch_size: Mini-batch size of training.
    :param max_steps: Maximum steps to run.  If the :param:`monitor` induces early-stopping, it might
                      take less steps than this number.
    :param shuffle: Whether or not to shuffle the data after each full-pass?
    :param summary_writer: If specified, will try to output the summary of training loss.
    """
    # check the arguments.
    monitor = _check_monitor(monitor)
    if not isinstance(train_data, DataFlow):
        train_data = TrainingBatchDataFlow(train_data, batch_size=batch_size, shuffle=shuffle)
    num_examples = train_data.num_examples
    if num_examples < batch_size:
        raise ValueError('Too few data such that no training step would be performed.')

    # restore the global step counter from the session.
    step_key = __name__ + '.run_steps:global_step'
    step = G.current_session().memo.get(step_key, 0)

    # prepare for the training.
    monitor.start_training(batch_size, num_examples // batch_size, max_steps)

    # the out loop indicates the pass of data (or to say, the epochs)
    epoch = 0
    while True:
        monitor.start_epoch(epoch)
        n_batches = 0
        total_loss = 0

        # the inner loop indicates the mini-batches of data.
        for args in train_data.iter_epoch():
            monitor.start_step(step)
            result = train_fn(*args)
            if isinstance(result, (tuple, list)):
                loss, summary = result[0], result[1]
            else:
                loss = result
                summary = None
            monitor.end_step(step, loss)

            # try to add the summary of training loss
            if summary is not None and summary_writer is not None:
                summary_writer.write(summary, global_step=step)

            n_batches += 1
            total_loss += loss
            step += 1
            G.current_session().memo[step_key] = step

            if step > max_steps or monitor.is_inducing_stopping:
                break

        monitor.end_epoch(epoch, float(total_loss) / n_batches)
        epoch += 1

        if step > max_steps or monitor.is_inducing_stopping:
            break

    # complete the training.
    monitor.end_training()
