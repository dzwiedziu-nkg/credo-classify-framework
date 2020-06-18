import pickle
import sys
from typing import List, TextIO, Callable, Optional, Tuple

from json import loads
from io import StringIO


LoadJsonCallback = Callable[[dict, int, List[dict]], Optional[bool]]


def load_json_from_stream(_input: TextIO, _filter: Optional[LoadJsonCallback] = None) -> Tuple[List[dict], int]:
    """
    Extract flat objects from array in JSON.

    Example::

      objects, count = load_json_from_stream(os.stdin, progress_load_filter)

    Example content of input JSON file::

      {
        "list": [
          {
            "key1": "value1",
            "key2": "value2",
            ...
          },
        ...]
      }

    How it works:
      1. Ignore all chars until ``'['``
      2. Extract string between next ``'{'`` and following ``'}'`` by:

         a) Ignore all chars until ``'{'``
         b) Copy all chars until ``'}'``

      3. Parse extracted string by JSON parser from stdlib.
      4. Execute filter if is not None, when is None or return True then append object to return list
      5. Go to 2. until ``']'``

    Note: depth of ``'{'`` was ignored, only flat object are supported

    :param _input: input text stream with JSON content

    :param _filter: optional callback function. Can be used for filter, progress notification,
      cancelling of read next and run some processes on parsed object.
      When is None then return effect is equivalent to return True by always.

    The ``_filter(obj, count, ret)`` callback provided as arg:
      Can be used for filter, progress notification and cancelling of read next.
      See ``progress_load_filter()`` or ``progress_and_process_image()`` for example how to implement custom callback method.

      Args:
        * ``obj``: parsed JSON object
        * ``count``: count of just parsed JSON object
        * ``ret``: list of just appended objects

      Return effect:
        * ``True``: parsed object will be append to ``ret`` list. Similar when ``_filter`` arg was not provided.
        * ``False``: object will be ignored (will not be append to ``ret`` list)
        * ``None``: object will be ignored and next object loop will be broken (cancel).

    :return: tuple of (list of appended objects, count of all parsed objects from input)
    """

    ret = []
    count = 0

    stage = 0
    buff = None
    for line in _input:
        done = False
        for a in line:
            if stage == 0:
                if a == '[':
                    stage = 1
                    continue  # and read next character
            if stage == 1:
                if a == ']':
                    done = True
                    break
                if a == '{':
                    buff = StringIO()
                    stage = 2  # and continue parsing this character in stage 2
            if stage == 2:
                if a == '}':
                    buff.write(a)
                    o = loads(buff.getvalue())
                    buff.close()
                    buff = None

                    count += 1
                    if filter is None:
                        ret.append(o)
                    else:
                        fr = _filter(o, count, ret)
                        if fr is None:
                            done = True
                            break
                        elif fr:
                            ret.append(o)
                    stage = 1
                else:
                    buff.write(a)
        if done:
            break

    return ret, count


def load_json(input_file: str, *args, **kwargs) -> Tuple[List[dict], int]:
    """
    Wrapper on ``load_json_from_stream()``.

    When ``input_file`` contains a ``"-"`` string then input will be read from ``stdin``.
    Otherwise the file will be open as input text stream.

    Examples::

      objects, count = load_json("-", progress_load_filter)
      objects, count = load_json("/tmp/detections.json", progress_and_process_image)

    :param input_file: path to JSON file or "-" for stdin.
    :return: redirected directly from load_json_from_stream()
    """

    inp = sys.stdin if input_file == '-' else open(input_file, 'r')
    ret = load_json_from_stream(inp, *args, **kwargs)
    if input_file != '-':
        inp.close()
    return ret


def serialize(output_file: str, obj_list: List[dict]) -> None:
    """
    Save data to binary file.

    Note: please refer to ``pickle`` module limitations.
    :param output_file: path to file when data will be stored
    :param obj_list: list of object to store
    """

    with open(output_file, 'wb') as f:
        pickle.dump(obj_list, f)


def deserialize(input_file: str) -> List[dict]:
    """
    Load data stored by ``serialize()``.

    :param input_file: path to file when data was stored by serialize()
    :return: list of objects
    """

    with open(input_file, "rb") as f:
        return pickle.load(f)
