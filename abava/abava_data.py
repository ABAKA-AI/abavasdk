# -*-coding:utf-8 -*-
import json


class ABAVA:
    def __init__(self, d=None):
        if isinstance(d, dict):
            for key, value in d.items():
                self.set_attr(key, value)
        elif isinstance(d, list):
            setattr(self, 'values', [ABAVA(value) for value in d])

    def __repr__(self):
        return 'molar_ai_format'

    def set_attr(self, k, v):
        if isinstance(v, dict):
            setattr(self, k, ABAVA(v))
        elif isinstance(v, list):
            for value in v:
                if isinstance(value, dict):
                    setattr(self, k, [ABAVA(value) for value in v])
                    break
                else:
                    setattr(self, k, v)
                    break
            if not v:
                setattr(self, k, [])
        elif isinstance(k, int):
            raise Exception("Cannot use data of type int as the key of a dictionary")
        else:
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)


def abava2dict(data):
    """
    Converting ABAVA types to dict
    :param data:
    :return:
    """
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                data[f'{k}'] = [abava2dict(d) for d in v]
            elif repr(v) == 'molar_ai_format':
                data[f'{k}'] = abava2dict(v)
    elif repr(data) == 'molar_ai_format':
        data = abava2dict(vars(data))
    return data


def gen_structure_json(data):
    """
    Writing dict to json files
    :param data:
    :return:
    """
    if isinstance(data, dict):
        with open('molar_ai_format.json', 'w') as f:
            json.dump(data, f, ensure_ascii=False)
    else:
        raise Exception('formatting error')
