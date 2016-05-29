# -*- coding: utf-8 -*-
import os
import re
from collections import OrderedDict

import six

from ipwxlearn.utils.concurrent import ThreadLocalStack
from ipwxlearn.utils.io import save_object_compressed, load_object_compressed
from ipwxlearn.utils.misc import silent_try
from .graph import current_graph

__all__ = ['BaseSession']


class CheckpointFile(object):
    """Class to read/write on a single checkpoint file."""

    def __init__(self, base_path, index):
        self.base_path = base_path
        self.index = index
        self._values_path = ''.join((self.base_path, '.v%d' % self.index))
        self._memo_path = ''.join((self.base_path, '.m%d' % self.index))

    def __lt__(self, other):
        return self.index < other.index

    def __cmp__(self, other):
        return self.index - other.index

    def __repr__(self):
        return 'CheckpointFile(%s,%d)' % (self.base_path, self.index)

    def read_values(self):
        if os.path.isfile(self._values_path):
            return load_object_compressed(self._values_path)

    def read_memo(self):
        if os.path.isfile(self._memo_path):
            return load_object_compressed(self._memo_path)

    @staticmethod
    def _safe_write(path, obj):
        p = ''.join((path, '.tmp'))
        try:
            save_object_compressed(p, obj)
            os.rename(p, path)
        except:
            silent_try(os.remove, p)
            raise

    def write_values(self, values):
        self._safe_write(self._values_path, values)

    def write_memo(self, memo):
        self._safe_write(self._memo_path, memo)

    def purge_values(self):
        if os.path.isfile(self._values_path):
            os.remove(self._values_path)

    @staticmethod
    def discover(base_path):
        parent, name = os.path.split(os.path.abspath(base_path))
        if not name:
            raise ValueError('%s is a directory rather than a file, which cannot be used as checkpoint.' %
                             repr(base_path))
        ret = []
        seen_indices = set()
        ext_pattern = re.compile(r'^\.[vm](\d+)$')
        for f in os.listdir(parent):
            fn, ext = os.path.splitext(f)
            if fn == name:
                m = ext_pattern.match(ext)
                if m:
                    idx = int(m.group(1))
                    if idx not in seen_indices:
                        ret.append(CheckpointFile(base_path, idx))
                        seen_indices.add(idx)
        ret.sort()
        return ret


class SessionMemo(object):
    """Dict-like object to read/write session memo."""

    def __init__(self):
        self._items = {}
        self._new_items = {}

    def __len__(self):
        return len(self._items)

    def __contains__(self, key):
        return key in self._items

    def __getitem__(self, key):
        return self._items[key]

    def __setitem__(self, key, value):
        self._items[key] = self._new_items[key] = value

    def get(self, key, default=None):
        return self._items.get(key, default)

    def items(self):
        return self._items.items()

    def values(self):
        return self._items.values()

    def keys(self):
        return self._items.keys()

    if six.PY2:
        def iteritems(self):
            return self._items.iteritems()

        def itervalues(self):
            return self._items.itervalues()

        def iterkeys(self):
            return self._items.iterkeys()

    def get_new(self):
        return self._new_items

    def clear_new(self):
        self._new_items.clear()


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

        # apart from the graph variable values, we also provide a session-wide resumable memo.
        self.memo = SessionMemo()

        # We preserve more than one checkpoint file in the whole session.
        #
        # The maximum number of checkpoint files is controlled by 'max_checkpoints',
        # while each checkpoint file takes the 'checkpoint_file' as the base name,
        # plus ".[index]" as the suffix.
        self._next_checkpoint = 1
        self._checkpoint_files = CheckpointFile.discover(checkpoint_file) if checkpoint_file else []
        if self._checkpoint_files:
            self._next_checkpoint = self._checkpoint_files[-1].index + 1

            # variable values should only be read from the latest checkpoint file.
            self.feed_values = self.feed_values or {}
            chk_values = self._checkpoint_files[-1].read_values()
            if chk_values:
                for k, v in six.iteritems(chk_values):
                    self.feed_values.setdefault(k, v)

            # memo values should be loaded from the entire history.
            for chk in self._checkpoint_files:
                chk_memo = chk.read_memo()
                if chk_memo:
                    for k, v in six.iteritems(chk_memo):
                        self.memo[k] = v
            self.memo.clear_new()

            # all right, now we could purge the stale checkpoint files.
            self._purge_stale_checkpoints()

    def _purge_stale_checkpoints(self):
        """Purge stale checkpoints from the directory."""
        if len(self._checkpoint_files) > self.max_checkpoints:
            purge_files = self._checkpoint_files[: -self.max_checkpoints]
            self._checkpoint_files = self._checkpoint_files[-self.max_checkpoints:]
            for chk in purge_files:
                silent_try(chk.purge_values)

    def checkpoint(self):
        """Make a checkpoint."""
        if not self.checkpoint_file:
            raise ValueError('Checkpoint file is not specified.')

        # write the checkpoint files
        var_dict = self.get_variable_values_dict(self.graph.get_variables(resumable=True))
        values = {
            self.graph.get_variable_info(var).full_name: value
            for var, value in six.iteritems(var_dict)
            }
        chk = CheckpointFile(self.checkpoint_file, self._next_checkpoint)
        try:
            chk.write_values(values)
            new_memo = self.memo.get_new()
            if new_memo:
                chk.write_memo(new_memo)
        except:
            chk.purge_values()
            raise
        self._checkpoint_files.append(chk)

        # increase the counter for next checkpoint.
        self._next_checkpoint += 1
        self.memo.clear_new()

        # purge stale checkpoints.
        self._purge_stale_checkpoints()

    @property
    def next_checkpoint_index(self):
        """Get the index of next checkpoint."""
        return self._next_checkpoint

    def __enter__(self):
        if self._has_entered_:
            raise ValueError('Session object is not reenterable.')

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
        self.graph.push_default()

        # finally, open the session.
        try:
            self._enter(feed_values, init_values)
        except:
            self.graph.pop_default()
            raise

        # push the session to stack.
        _session_stack.push(self)
        self._has_entered_ = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            # close the session.
            last_values = self._exit(self.graph.get_persistent_variables())
            self.graph.set_last_values(last_values)

        finally:
            # exit the session context.
            _session_stack.pop()

            # exit the graph context.
            self.graph.pop_default()

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

    def get_variable_values(self, vars):
        """
        Get the values of specified variables.

        :param vars: Backend variable, or iterable of backend variable objects
        :return: Value of the single variable, or tuple of variable values.
        """
        raise NotImplementedError()

    def set_variable_values(self, vars_values):
        """
        Set the values of specified variables.

        :param vars_values: Dict from backend variables to their values.
        """
        raise NotImplementedError()

    def get_variable_values_dict(self, vars):
        """
        Get the values of specified variables as dict.

        :param vars: iterable backend variable objects
        :return: dict from backend variable objects to their values.
        """
        vars = list(vars)
        values = self.get_variable_values(vars)
        ret = OrderedDict()
        for var, value in zip(vars, values):
            ret[var] = value
        return ret


#: Thread local session stack, with a default session on the thread.
_session_stack = ThreadLocalStack()


def current_session():
    """
    Get the current active session.
    :rtype: :class:`BaseSession`
    """
    if _session_stack.empty:
        raise ValueError('No session is activated.')
    return _session_stack.top
