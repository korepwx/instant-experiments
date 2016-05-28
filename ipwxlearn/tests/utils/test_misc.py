# -*- coding: utf-8 -*-
import unittest

from ipwxlearn.utils import misc


class MiscTestCase(unittest.TestCase):

    def test_assert_raises_message(self):
        """Test the helper assert function that checks the exception message."""
        with misc.assert_raises_message(self, ValueError, "This is value error."):
            raise ValueError("This is value error.")

        with self.assertRaises(AssertionError) as cm:
            with misc.assert_raises_message(self, TypeError, "This is type error."):
                v = 1 + 2
        self.assertEquals(str(cm.exception), 'TypeError not raised')

        with self.assertRaises(AssertionError) as cm:
            with misc.assert_raises_message(self, ValueError, "This is value error."):
                raise ValueError("not valid message")
        self.assertTrue(str(cm.exception).startswith("'not valid message' != 'This is value error.'"))
