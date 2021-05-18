import json
import pickle
import sys
from typing import List, TextIO, Callable, Optional, Tuple, Any

from json import loads
from io import StringIO

from credo_cf import METADATA_MAX, METADATA_AVERAGE, METADATA_BLACKS, METADATA_BLACKS_THRESHOLD, METADATA_AX, METADATA_AY, METADATA_AZ, METADATA_ORIENTATION, \
    METADATA_TEMPERATURE, METADATA_PARSED

LoadJsonCallback = Callable[[dict, int, List[dict]], Optional[bool]]


def load_json_from_stream(_input: TextIO, _parser: Optional[LoadJsonCallback] = None) -> Tuple[List[dict], int, List[str]]:
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

    :param _parser: optional callback function. Can be used for filter, progress notification,
      cancelling of read next and run some processes on parsed object.
      When is None then return effect is equivalent to return True by always.

    The ``_parser(obj, count, ret)`` callback provided as arg:
      Can be used for filter, progress notification and cancelling of read next.
      See ``progress_load_filter()`` or ``progress_and_process_image()`` for example how to implement custom callback method.

      Args:
        * ``obj``: parsed JSON object
        * ``count``: count of just parsed JSON object
        * ``ret``: list of just appended objects

      Return effect:
        * ``True``: parsed object will be append to ``ret`` list. Similar when ``_parser`` arg was not provided.
        * ``False``: object will be ignored (will not be append to ``ret`` list)
        * ``None``: object will be ignored and next object loop will be broken (cancel).

    :return: tuple of (list of appended objects, count of all parsed objects from input, errors of parsed)
    """

    ret = []
    count = 0
    errors = []

    stage = 0
    buff = None
    for line in _input:
        done = False
        in_quotes = False
        next_ignore = False
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
                if next_ignore:
                    next_ignore = False
                if a == '\\':
                    next_ignore = True
                if a == '"' and not next_ignore:
                    in_quotes = not in_quotes

                if a == '}' and not in_quotes:
                    if buff is None:
                        errors.append('invalid stage, please review this file in debugger')
                        buff = StringIO()
                        stage = 1
                        continue

                    buff.write(a)
                    obj_json = buff.getvalue()
                    buff.close()
                    buff = None
                    try:
                        o = loads(obj_json)

                        if len(o.get('metadata', '')) > 0:
                            try:
                                metadata = loads(o.get('metadata'))
                                o[METADATA_MAX] = metadata.get('max')
                                o[METADATA_AVERAGE] = metadata.get('average')
                                o[METADATA_BLACKS] = metadata.get('blacks')
                                o[METADATA_BLACKS_THRESHOLD] = metadata.get('black_threshold')
                                o[METADATA_AX] = metadata.get('ax')
                                o[METADATA_AY] = metadata.get('ay')
                                o[METADATA_AZ] = metadata.get('az')
                                o[METADATA_ORIENTATION] = metadata.get('orientation')
                                o[METADATA_TEMPERATURE] = metadata.get('temperature')
                                o[METADATA_PARSED] = True
                                o['metadata'] = ''
                            except:
                                pass

                        count += 1
                        if _parser is None:
                            ret.append(o)
                        else:
                            fr = _parser(o, count, ret)
                            if fr is None:
                                done = True
                                break
                            elif fr:
                                ret.append(o)
                        stage = 1
                    except:
                        errors.append(obj_json)
                        stage = 1
                else:
                    buff.write(a)
        if done:
            break

    return ret, count, errors


def load_json(input_file: str, *args, **kwargs) -> Tuple[List[dict], int, List[str]]:
    """
    Wrapper on ``load_json_from_stream()``.

    When ``input_file`` is the ``"-"`` string then input will be read from ``stdin``.
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


def save_json_to_stream(detections: List[dict], *args, **kwargs):
    """
    Write detections list to JSON file.

    :param detections: hits to save in JSON file
    :param args: unnamed args redirected to json.dump
    :param kwargs: name args redirected to json.dump
    """
    json.dump({'detections': detections}, *args, **kwargs)


def save_json(detections: List[dict], output_file: str = '-', *args, **kwargs):
    """
    Wrapper on ``save_json_to_stream``.

    When ``output_file`` is the ``"-"`` string then output will be write to ``stdout``.
    Otherwise the file will be open as output text stream.

    Examples::
      save_json('/tmp/output.json', detections, ident=2)

    :param output_file: path to output JSON file or "-" for stdout.
    :param detections: hits to save in JSON
    :param args: unnamed args redirected to json.dump except second arg
    :param kwargs: unnamed args redirected to json.dump except ``fp``
    :return: None
    """
    f = sys.stdout if output_file == '-' else open(output_file, 'w')
    save_json_to_stream(detections, f, *args, **kwargs)
    if output_file != '-':
        f.close()


def serialize(output_file: str, obj_list: Any) -> None:
    """
    Save data to binary file.

    Note: please refer to ``pickle`` module limitations.
    :param output_file: path to file when data will be stored
    :param obj_list: list of object to store
    """

    with open(output_file, 'wb') as f:
        pickle.dump(obj_list, f)


def deserialize(input_file: str) -> Any:
    """
    Load data stored by ``serialize()``.

    :param input_file: path to file when data was stored by serialize()
    :return: list of objects
    """

    with open(input_file, "rb") as f:
        return pickle.load(f)


def deserialize_or_run(input_file: str, compute_function: Optional[Callable[[Any], Any]], *args, **kwargs) -> Any:
    """
    Try to deserialize object from ``input_file`` or get result from ``compute_function`` executed with ``param``.

    :param input_file: path to file name with serialized data
    :param compute_function: launch when input_file not found
    :return: deserialized object from input_file or result of compute_function
    """

    import os.path

    if os.path.isfile(input_file):
        return deserialize(input_file)
    d = compute_function(*args, **kwargs)
    serialize(input_file, d)
    return d
