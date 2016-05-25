# -*- coding: utf-8 -*-
import six

from ipwxlearn.utils.concurrent import ThreadLocalStack


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
    :param feed_values: Dict of values that should be used to initialize variables for this session.
                        Keys of this dict should be either variable full names, or the backend variable
                        objects.
    :param init_variables: If True, will re-init the variables not specified in :param:`feed_values` with
                           corresponding initializers.  If False, will restore the values saved from last
                           session.
    """

    #: Indicate whether or not the session has been entered and exited.
    _has_exited_ = False

    def __init__(self, graph, feed_values=None, init_variables=False):
        self.graph = graph
        self.feed_values = feed_values
        self.init_variables = init_variables

    def __enter__(self):
        if self._has_exited_:
            raise ValueError('Could not enter the session, since it has been exited.')

        # merge feed values from all sources.
        from .graph import current_graph, VariableTags
        feed_values = {}
        if self.feed_values is not None:
            for k, v in six.iteritems(self.feed_values):
                var = current_graph().get_variable(k) if isinstance(k, six.string_types) else v
                feed_values[var] = v

        last_values = current_graph().get_last_values(current_graph().iter_variables(tags=[VariableTags.PERSISTENT]))
        for var, value in six.iteritems(last_values):
            if var not in feed_values:
                feed_values[var] = value

        # get the initializers for each variable.
        init_values = {}
        for var, info in six.iteritems(current_graph().variable_info_dict):
            if var not in feed_values and info is not None:
                init_values[var] = info

        # finally, open the session.
        self._enter(feed_values, init_values)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from .graph import current_graph, VariableTags
        if not self._has_exited_:
            self._has_exited_ = True
            last_values = self._exit(current_graph().iter_variables(tags=[VariableTags.PERSISTENT]))
            current_graph().set_last_values(last_values)

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


#: Thread local session stack, with a default session on the thread.
_session_stack = ThreadLocalStack()


def current_session():
    """
    Get the current active session.
    :rtype: :class:`BaseSession`
    """
    return _session_stack.top
