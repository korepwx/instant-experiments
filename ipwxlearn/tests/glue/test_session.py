# -*- coding: utf-8 -*-
import unittest

from ipwxlearn.glue import G


class SessionTestCase(unittest.TestCase):

    def test_session_vars(self):
        """Test save/restore/init variables for a session."""

        with G.Graph().as_default() as graph:
            pass
