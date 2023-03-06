import json
import pandas


def from_file(path: str):
    return pandas.read_json(path)
    with open(path, encoding="utf-8", mode='r') as f_in:
        return json.load(f_in)
