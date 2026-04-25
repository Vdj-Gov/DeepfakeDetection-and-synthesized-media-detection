import sys, os, importlib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    mod = importlib.import_module('models.xceptionModel')
    print('module file:', mod.__file__)
    print('has createXceptionModel:', hasattr(mod, 'createXceptionModel'))
    print('keys:', [k for k in mod.__dict__.keys() if not k.startswith('__')])
except Exception as e:
    import traceback
    traceback.print_exc()
