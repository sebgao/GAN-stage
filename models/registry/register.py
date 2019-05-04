class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def _register_module(self, module_class):
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def __getitem__(self, name):
        return self._module_dict[name]

    def register_module(self, cls):
        self._register_module(cls)
        return cls

class Libray:
    def __init__(self, dicts):
        self._dicts = dicts
    
    def __getattr__(self, name):
        for _dict in self._dicts:
            if name in _dict._module_dict:
                return _dict[name]
        return None

GENERATOR = Registry('generator')
DISCRIMINATOR = Registry('discriminator')
LOSS = Registry('loss')
LIBRARY = Libray((GENERATOR, DISCRIMINATOR, LOSS))
