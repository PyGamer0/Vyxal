import os
import sys

# Pipped modules

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.insert(1, THIS_FOLDER)


class Context:
    """An instance of this class holds execution variables set by flags"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all variables"""
        print("reset lol")
        self.context_level = 0
        self.context_values = [0]
        self.global_stack = []
        self.input_level = 0
        self.inputs = []
        self.input_values = {0: [self.inputs, 0]}  # input_level: [source, input_index]
        self.last_popped = []
        self.keg_mode = False
        self.number_iterable = list
        self.raw_strings = False
        self.online_version = False
        self.output = None
        self.printed = False
        self.register = 0
        self.retain_items = False
        self.reverse_args = False
        self.safe_mode = (
            False  # You may want to have safe evaluation but not be online.
        )
        self.stack = []
        self.variables_are_digraphs = False

        self.MAP_START = 1
        self.MAP_OFFSET = 1
        self._join = False
        self._vertical_join = False
        self.use_encoding = False
        self.stderr = None

    def this_function(self, x):
        from vyxal.builtins import vy_print

        vy_print(self.stack, ctx=self)
        return x

    def set_globals(self, flags):
        """Set globals according to given flags"""
        if "H" in flags:
            self.stack = [100]

        if "a" in flags:
            self.inputs = [self.inputs]

        if "M" in flags:
            self.MAP_START = 0

        if "m" in flags:
            self.MAP_OFFSET = 0

        if "Ṁ" in flags:
            self.MAP_START = 0
            self.MAP_OFFSET = 0

        if "R" in flags:
            self.number_iterable = range

        self._join = "j" in flags
        self._vertical_join = "L" in flags
        self.use_encoding = "v" in flags
        self.reverse_args = "r" in flags
        self.keg_mode = "K" in flags
        self.safe_mode = "E" in flags
        self.raw_strings = "D" in flags
