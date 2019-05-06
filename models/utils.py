from .registry import LIBRARY

def instantiate(dict_class):
    if callable(dict_class):
        return dict_class()
        
    if isinstance(dict_class, dict):
        assert 'type' in dict_class
        _type = dict_class.pop('type')
        #assert _type in LIBRARY
        return LIBRARY[_type](**dict_class)