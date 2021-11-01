'''
Semantic Versioning
---
MAJOR: incompatible changes in model IO schema
MINOR: backward compatible changes in model IO schema
PATCH: changes in model training  procedure
'''
VERSION = (0, 1, 0)

__version__ = '.'.join(map(str, VERSION))