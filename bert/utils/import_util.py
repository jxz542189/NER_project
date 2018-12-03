import importlib
from bert.utils import ConfigError
def cls_from_str(name: str) -> type:
    """Returns a class object with the name given as a string."""
    try:
        module_name, property_name = name.split(':')
        module_names = module_name.split('.')
        module_name, cls_name = '.'.join(module_names[:-1]), module_names[-1]
    except ValueError:
        raise ConfigError('Expected class description in a `module.submodules:ClassName` form, but got `{}`'
                          .format(name))

    return getattr(getattr(importlib.import_module(module_name), cls_name), property_name)