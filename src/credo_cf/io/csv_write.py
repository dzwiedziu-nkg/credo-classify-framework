import csv
from typing import List


def to_str(v: any):
    if v is None:
        return ''
    return str(v)


def gen_csv_header(objects: List[dict]) -> dict:
    """
    Make options for CVS export.
    :param objects: parsed objects to analyse.
    :return: key: column name, value:
    - type: trivial or tuple
    - count: when tuple then max count values in tuple
    """
    options = {}
    for o in objects:
        for k, v in o.items():
            if isinstance(v, tuple):
                option = options.get(k, {'type': 'tuple', 'count': len(v)})
                option['count'] = max(option['count'], len(v))
                options[k] = option
            if isinstance(v, (str, int, float)):
                options[k] = {'type': 'trivial'}
    return options


def write_to_csv(csvfile, objects: List[dict], options: dict = None, header: List[str] = None, exclude: set = None, regex_exclude: List = None) -> None:
    """

    :param csvfile: file stream to output in CSV format
    :param objects: objects to write to CSV
    :param options: options for CSV columns, see: gen_csv_header
    :param header: optional, columns to write
    :param exclude: exclude columns
    :param regex_exclude: exclude columns regex
    """
    writer = csv.writer(csvfile)
    _header = header or sorted(options.keys())
    _options = options or {}
    _exclude = exclude or set()
    _regex_exclude = regex_exclude or []

    def in_exclude(col: str):
        if col in _exclude:
            return True
        for r in _regex_exclude:
            if r.match(col):
                return True
        return False

    header_row = []
    for h in _header:
        if in_exclude(h):
            continue

        opt = _options.get(h, {})
        header_row.append(h)
        if opt.get('type') == 'tuple':
            for i in range(1, opt.get('count', 1)):
                header_row.append('')
    writer.writerow(header_row)

    for o in objects:
        row = []
        for h in _header:
            if in_exclude(h):
                continue

            opt = _options.get(h, {})
            _type = opt.get('type', 'trivial')
            count = opt.get('count', 1)
            v = o.get(h)
            if _type == 'trivial':
                row.append(to_str(v))
            elif _type == 'tuple':
                v = v or ()
                for t in v:
                    row.append(to_str(t))
                for i in range(len(v), count):
                    row.append('')
        writer.writerow(row)
