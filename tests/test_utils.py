import os
import sys
from multiprocessing import Manager

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.insert(1, THIS_FOLDER)

from vyxal.vy_globals import CTX
from vyxal import interpreter as interp

header = "stack = []\nregister = 0\nprinted = False\n"
manager = Manager()


def run_code(code, flags="", input_list=[], output_variable=manager.dict()):
    reset_globals()
    interp.execute(code, flags, "\n".join(map(str, input_list)), output_variable)
    return CTX.stack


def reset_globals():
    CTX.stack = []
    interp.stack = []
    CTX.context_level = 0
    CTX.context_values = [0]
    CTX.global_stack = []
    CTX.input_level = 0
    CTX.inputs = []
    CTX.input_values = {
        0: [CTX.inputs, 0]
    }  # input_level: [source, input_index]
    CTX.last_popped = []
    CTX.keg_mode = False
    CTX.number_iterable = list
    CTX.raw_strings = False
    CTX.online_version = False
    CTX.output = ""
    CTX.printed = False
    CTX.register = 0
    CTX.retain_items = False
    CTX.reverse_args = False
    CTX.safe_mode = (
        False  # You may want to have safe evaluation but not be online.
    )
    CTX.stack = []
    CTX.variables_are_digraphs = False

    CTX.MAP_START = 1
    CTX.MAP_OFFSET = 1
    CTX._join = False
    CTX._vertical_join = False
    CTX.use_encoding = False
    CTX.set_globals("")


def reshape(arr, shape):
    if len(shape) == 1:
        return arr
    rest = shape[1:]
    size = len(arr) // shape[0]
    return [reshape(arr[i * size : (i + 1) * size], rest) for i in range(shape[0])]


def to_list(vector):
    typ = interp.vy_type(vector)
    if typ in (list, interp.Generator):
        return list(
            map(to_list, vector._dereference() if typ is interp.Generator else vector)
        )
    return vector
