# -*- coding: utf-8 -*-
import os
import re

import six

from ipwxlearn.glue.common.graph import VariableTags, current_graph
from ipwxlearn.utils.concurrent import ThreadLocalStack
from ipwxlearn.utils.io import save_object_compressed, load_object_compressed
from ipwxlearn.utils.misc import silent_try


class BaseSession(object):
    """
    Base class for all tensor computing session.

    Various backend may deal with variables in different ways.  For example, Theano variables are strongly
    tied to some space on the device memory.  While on the other hand, TensorFlow variables are just sort
    of descriptions of these data, having dedicated spaces for each different session.

    Due to the limitations of these different types of backend, this session class cannot guarantee that
    variables in different sessions would have dedicated storage spaces.  However, the session classes
    provide a unique interface to initialize these variables at the beginning of each session, and to
    save the values of these variables into graph instances after a session has done.  This would make
    it possible to write code running on different backend more easily.

    By design, sessions are intended to be used as context manager, while each session should be entered
    for exactly once.

    :param graph: Graph instance that holds all the variables needed in the session.
                  This graph will be set to default graph once the session is entered.
                  If not specified, use the current graph.
    :param feed_values: Dict of values that should be used to initialize variables for this session.
                        Keys of this dict should be either variable full names, or the backend variable
                        objects.
    :param checkpoint_file: If specified, will load the variables from external checkpoint file.
                            Also, calling :method:`checkpoint` will save the variables to that file.
    :param init_variables: If True, will re-init the variables not specified in :param:`feed_values` with
                           corresponding initializers.  If False, will restore the values saved from last
                           session.
    """

    #: Indicate whether or not the session has been entered.
    _has_entered_ = False

    def __init__(self, graph=None, feed_values=None, init_variables=False, checkpoint_file=None,
                 max_checkpoints=10):
        self.graph = graph or current_graph()
        self.feed_values = feed_values
        self.init_variables = init_variables
        self.checkpoint_file = checkpoint_file
        self.max_checkpoints = max_checkpoints

        # entered graph context manager
        self._graph_ctx = None

        # We preserve more than one checkpoint file in the whole session.
        #
        # The maximum number of checkpoint files is controlled by 'max_checkpoints',
        # while each checkpoint file takes the 'checkpoint_file' as the base name,
        # plus ".[index]" as the suffix.
        self._next_checkpoint = 1
        self._checkpoint_files = []

        # load the checkpoint file.
        if self.checkpoint_file is not None:
            self._discover_checkpoints()
        if self._checkpoint_files:
            self._next_checkpoint = self._checkpoint_files[-1][0] + 1
            self._load_checkpoint_file(self._checkpoint_files[-1][1])

    def _discover_checkpoints(self):
        """Discover the checkpoint files."""
        parent, name = os.path.split(os.path.abspath(self.checkpoint_file))
        if not name:
            raise ValueError('%s is a directory rather than a file, which cannot be used as checkpoint.' %
                             repr(self.checkpoint_file))
        ext_pattern = re.compile(r'^\.\d+$')
        for f in os.listdir(parent):
            fn, ext = os.path.splitext(f)
            if fn == name and ext_pattern.match(ext):
                idx = int(ext[1:])
                self._checkpoint_files.append((idx, os.path.join(parent, f)))
        self._checkpoint_files.sort()

    def _purge_stale_checkpoints(self):
        """Purge stale checkpoints from the directory."""
        if len(self._checkpoint_files) > self.max_checkpoints:
            purge_files = self._checkpoint_files[: -self.max_checkpoints]
            self._checkpoint_files = self._checkpoint_files[-self.max_checkpoints:]
            for _, fn in purge_files:
                silent_try(os.remove, fn)

    def _load_checkpoint_file(self, path):
        """Load the specified checkpoint file."""
        states = load_object_compressed(path)
        if self.feed_values is None:
            self.feed_values = {}
        for k, v in six.iteritems(states['values']):
            self.feed_values.setdefault(self.graph.get_variable(k), v)

    def _save_checkpoint_file(self, path):
        """Save the values to specified checkpoint file."""
        var_dict = self._extract_vars(self.graph.get_variables(tags=[VariableTags.RESUMABLE]))
        states = {
            'values': {
                self.graph.get_variable_info(var).full_name: value
                for var, value in six.iteritems(var_dict)
            }
        }
        # to prevent a partially saved checkpoint file, we first save to a temporary file,
        # then rename it to the final checkpoint file.
        tmpfile = '%s.tmp' % path
        try:
            save_object_compressed(tmpfile, states)
            os.rename(tmpfile, path)
        finally:
            silent_try(os.remove, tmpfile)

    def checkpoint(self):
        """Make a checkpoint."""
        if not self.checkpoint_file:
            raise ValueError('Checkpoint file is not specified.')
        path = '%s.%s' % (self.checkpoint_file, self._next_checkpoint)
        self._save_checkpoint_file(path)
        self._next_checkpoint += 1
        self._checkpoint_files.append(path)
        self._purge_stale_checkpoints()

    @property
    def next_checkpoint_index(self):
        """Get the index of next checkpoint."""
        return self._next_checkpoint

    def __enter__(self):
        if self._has_entered_:
            raise ValueError('Session object is not reenterable..')

        # merge feed values from all sources.
        feed_values = {}
        if self.feed_values is not None:
            for k, v in six.iteritems(self.feed_values):
                var = self.graph.get_variable(k) if isinstance(k, six.string_types) else k
                feed_values[var] = v

        if not self.init_variables:
            last_values = self.graph.get_last_values_as_dict(self.graph.get_persistent_variables())
            for var, value in six.iteritems(last_values):
                if var not in feed_values:
                    feed_values[var] = value

        # get the initializers for each variable.
        init_values = {}
        for var, info in six.iteritems(self.graph.variable_info_dict):
            if var not in feed_values and info is not None:
                init_values[var] = info.init

        # set the graph as the default graph.
        self._graph_ctx = self.graph.as_default()
        self._graph_ctx.__enter__()

        # finally, open the session.
        self._enter(feed_values, init_values)
        _session_stack.push(self)
        self._has_entered_ = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # close the session.
        last_values = self._exit(self.graph.get_persistent_variables())
        self.graph.set_last_values(last_values)
        _session_stack.pop()

        # exit the graph context.
        self._graph_ctx.__exit__(exc_type, exc_val, exc_tb)

    def _enter(self, feed_values, init_values):
        """
        Enter the session for computing.
        Derived classes might override this to actually initialize the backend variables and open the backend session.

        :param feed_values: Dict of values that should be used to initialize variables for this session.
                            Keys of this dict should be the backend variable objects.
        :param init_values: Dict of initializers that should be used to init variables for this session.
                            Keys of this dict should be the backend variable objects.
        """
        raise NotImplementedError()

    def _exit(self, save_vars):
        """
        Exit the session, returning the values for variables that should be saved to graph.
        Derived classes might override this to save variables from the backend session, and release session resources.

        :param save_vars: iterable backend variable objects, whose values should be saved from the session.
        :return: dict from backend variable objects to their values.
        """
        raise NotImplementedError()

    def _extract_vars(self, vars):
        """
        Extract values of variables from a running session.

        :param vars: iterable backend variable objects
        :return: dict from backend variable objects to their values.
        """
        raise NotImplementedError()


#: Thread local session stack, with a default session on the thread.
_session_stack = ThreadLocalStack()


def current_session():
    """
    Get the current active session.
    :rtype: :class:`BaseSession`
    """
    return _session_stack.top
