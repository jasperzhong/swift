from typing import List, Callable, Dict, Optional, Any
import builtins
import sys
import importlib
from torch.serialization import _load
import pickle
import torch
import _compat_pickle
import types
import os.path

from ._importlib import _normalize_line_endings, _resolve_name, _sanity_check, _calc___package__, \
                        _zip_searchorder, _normalize_path
from ._mock_zipreader import MockZipReader

class HermeticImporter:
    """Importers allow you to load code written to packages by HermeticExporter.
    Code is loaded in a hermetic way, using files from the package
    rather than the normal python import system. This allows
    for the packaging of PyTorch model code and data so that it can be run
    on a server or used in the future for transfer learning.

    The importer for packages ensures that code in the module can only be loaded from
    within the package, except for modules explicitly listed as external during export.
    The file `extern_modules` in the zip archive lists all the modules that a package externally depends on.
    This prevents "implicit" dependencies where the package runs locally because it is importing
    a locally-installed package, but then fails when the package is copied to another machine.
    """

    modules : Dict[str, Optional[types.ModuleType]]
    """The dictionary of already loaded modules from this package, equivalent to `sys.modules` but
    local to this importer.
    """

    def __init__(self, filename: str, module_allowed: Callable[[str], bool] = lambda module_name: True):
        """Open `filename` for importing. This checks that the imported package only requires modules
        allowed by `module_allowed`

        Args:
            filename (str): archive to load. Can also be a directory of the unzipped files in the archive
                for easy debugging and editing.
            module_allowed (Callable[[str], bool], optional): A method to determine if a externally provided module
                should be allowed. Can be used to ensure packages loaded do not depend on modules that the server
                does not support. Defaults to allowing anything.

        Raises:
            ImportError: If the package will use a disallowed module.
        """
        self.filename = filename
        self.zip_reader = torch._C.PyTorchFileReader(self.filename) if not os.path.isdir(self.filename) else MockZipReader(self.filename)
        self.files = _get_all_files(self.zip_reader.get_all_records())
        self.modules = {}
        self.extern_modules = self._read_extern()
        for extern_module in self.extern_modules:
            if not module_allowed(extern_module):
                raise ImportError(f"package '{filename}' needs the external module '{extern_module}' but that module has been disallowed")

        self.patched_builtins = builtins.__dict__.copy()
        self.patched_builtins['__import__'] = self.__import__
        self.modules['resources'] = self # allow pickles from archive using `import resources`

        # used for torch.serialization._load
        self.Unpickler = lambda *args, **kwargs: _UnpicklerWrapper(self, *args, **kwargs)

    def import_module(self, name: str, package=None) -> types.ModuleType:
        """Load a module from the package if it hasn't already been loaded, and then return
        the module. Modules are loaded locally
        to the importer and will appear in `self.modules` rather than `sys.modules`

        Args:
            name (str): Fully qualified name of the module to load.
            package ([type], optional): Unused, but present to match the signature of importlib.import_module. Defaults to None.

        Returns:
            types.ModuleType: the (possibly already) loaded module.
        """
        return self._gcd_import(name)

    def load_binary(self, package: str, resource: str) -> bytes:
        """Load raw bytes.

        Args:
            package (str): The name of module package (e.g. "my_package.my_subpackage")
            resource (str): The unique name for the resource.

        Returns:
            bytes: The loaded data.
        """

        path = self._zipfile_path(package, resource)
        return self.zip_reader.get_record(path)

    def load_text(self, package: str, resource: str,  encoding:str ='utf-8', errors:str='strict') -> str:
        """Load a string.

        Args:
            package (str): The name of module package (e.g. "my_package.my_subpackage")
            resource (str): The unique name for the resource.
            encoding (str, optional): Passed to `decode`. Defaults to 'utf-8'.
            errors (str, optional): Passed to `decode`. Defaults to 'strict'.

        Returns:
            str: The loaded text.
        """
        data = self.load_binary(package, resource)
        return data.decode(encoding, errors)

    def load_pickle(self, package: str, resource: str, map_location=None) -> Any:
        """Unpickles the resource from the package, loading any modules that are needed to construct the objects
        using :meth:`import_module`

        Args:
            package (str): The name of module package (e.g. "my_package.my_subpackage")
            resource (str): The unique name for the resource.
            map_location: Passed to `torch.load` to determine how tensors are mapped to devices. Defaults to None.

        Returns:
            Any: the unpickled object.
        """
        pickle_file = self._zipfile_path(package, resource)
        return _load(self.zip_reader, map_location, self, pickle_file=pickle_file)


    def _read_extern(self):
        return self.zip_reader.get_record('extern_modules').decode('utf-8').splitlines(keepends=False)

    def _make_module(self, name):
        code, filename, is_package = self._get_module_info(name)
        spec = importlib.machinery.ModuleSpec(name, self, is_package=is_package)
        module = importlib.util.module_from_spec(spec)
        self.modules[name] = module
        ns = module.__dict__
        ns['__spec__'] = spec
        ns['__loader__'] = self
        ns['__file__'] = filename
        ns['__cached__'] = None
        ns['__builtins__'] = self.patched_builtins
        if code is not None:
            exec(code, ns)
        return module

    # Get the code object associated with the module specified by
    # 'fullname'.
    def _get_module_info(self, name):
        path = name.replace('.', '/')
        for suffix, ispackage in _zip_searchorder:
            fullpath = path + suffix
            if fullpath in self.files and self.files[fullpath] == 'regular':
                # its a file in the archive, and not a directory, this is a module
                return self._compile_source(fullpath), fullpath, ispackage

        if path in self.files and self.files[path] == 'directory':
            # this is a namespace module
            return None, None, True

        raise ModuleNotFoundError(f'No module named "{name}" in self-contained archive "{self.filename}" and the module is also not in the list of allowed external modules: {self.extern_modules}')

    def _compile_source(self, fullpath):
        source = self.zip_reader.get_record(fullpath)
        source = _normalize_line_endings(source)
        return compile(source, fullpath, 'exec', dont_inherit=True)

    def source_for_module(self, module) -> str:
        assert module.__loader__ is self
        return self.zip_reader.get_record(module.__file__).decode('utf-8')

    # note: copied from cpython's import code, with call to create module replaced with _make_module
    def _do_find_and_load(self, name):
        path = None
        parent = name.rpartition('.')[0]
        if parent:
            if parent not in self.modules:
                self._gcd_import(parent)
            # Crazy side-effects!
            if name in self.modules:
                return self.modules[name]
            parent_module = self.modules[parent]
            try:
                path = parent_module.__path__
            except AttributeError:
                msg = (_ERR_MSG + '; {!r} is not a package').format(name, parent)
                raise ModuleNotFoundError(msg, name=name) from None

        module = self._make_module(name)

        if parent:
            # Set the module as an attribute on its parent.
            parent_module = self.modules[parent]
            setattr(parent_module, name.rpartition('.')[2], module)
        return module

    # note: copied from cpython's import code
    def _find_and_load(self, name):
        module = self.modules.get(name, _NEEDS_LOADING)
        if module is _NEEDS_LOADING:
            return self._do_find_and_load(name)

        if module is None:
            message = ('import of {} halted; '
                    'None in sys.modules'.format(name))
            raise ModuleNotFoundError(message, name=name)

        return module


    def _gcd_import(self, name, package=None, level=0):
        """Import and return the module based on its name, the package the call is
        being made from, and the level adjustment.

        This function represents the greatest common denominator of functionality
        between import_module and __import__. This includes setting __package__ if
        the loader did not.

        """
        _sanity_check(name, package, level)
        if level > 0:
            name = _resolve_name(name, package, level)


        root_name = name.split('.')[0]
        if root_name in self.extern_modules:
            # external dependency, use the normal import system
            return importlib.import_module(name, package)

        return self._find_and_load(name)

    # note: copied from cpython's import code
    def _handle_fromlist(self, module, fromlist, *, recursive=False):
        """Figure out what __import__ should return.

        The import_ parameter is a callable which takes the name of module to
        import. It is required to decouple the function from assuming importlib's
        import implementation is desired.

        """
        # The hell that is fromlist ...
        # If a package was imported, try to import stuff from fromlist.
        if hasattr(module, '__path__'):
            for x in fromlist:
                if not isinstance(x, str):
                    if recursive:
                        where = module.__name__ + '.__all__'
                    else:
                        where = "``from list''"
                    raise TypeError(f"Item in {where} must be str, "
                                    f"not {type(x).__name__}")
                elif x == '*':
                    if not recursive and hasattr(module, '__all__'):
                        self._handle_fromlist(module, module.__all__,
                                        recursive=True)
                elif not hasattr(module, x):
                    from_name = '{}.{}'.format(module.__name__, x)
                    try:
                        self._gcd_import(from_name)
                    except ModuleNotFoundError as exc:
                        # Backwards-compatibility dictates we ignore failed
                        # imports triggered by fromlist for modules that don't
                        # exist.
                        if (exc.name == from_name and
                            self.modules.get(from_name, _NEEDS_LOADING) is not None):
                            continue
                        raise
        return module

    def __import__(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0:
            module = self._gcd_import(name)
        else:
            globals_ = globals if globals is not None else {}
            package = _calc___package__(globals_)
            module = self._gcd_import(name, package, level)
        if not fromlist:
            # Return up to the first dot in 'name'. This is complicated by the fact
            # that 'name' may be relative.
            if level == 0:
                return self._gcd_import(name.partition('.')[0])
            elif not name:
                return module
            else:
                # Figure out where to slice the module's name up to the first dot
                # in 'name'.
                cut_off = len(name) - len(name.partition('.')[0])
                # Slice end needs to be positive to alleviate need to special-case
                # when ``'.' not in name``.
                return self.modules[module.__name__[:len(module.__name__)-cut_off]]
        else:
            return self._handle_fromlist(module, fromlist)

    def _get_package(self, package):
        """Take a package name or module object and return the module.

        If a name, the module is imported.  If the passed or imported module
        object is not a package, raise an exception.
        """
        if hasattr(package, '__spec__'):
            if package.__spec__.submodule_search_locations is None:
                raise TypeError('{!r} is not a package'.format(
                    package.__spec__.name))
            else:
                return package
        else:
            module = self.import_module(package)
            if module.__spec__.submodule_search_locations is None:
                raise TypeError('{!r} is not a package'.format(package))
            else:
                return module

    def _zipfile_path(self, package, resource):
        package = self._get_package(package)
        resource = _normalize_path(resource)
        assert package.__loader__ is self
        return f"{package.__name__.replace('.', '/')}/{resource}"

_NEEDS_LOADING = object()
_ERR_MSG_PREFIX = 'No module named '
_ERR_MSG = _ERR_MSG_PREFIX + '{!r}'

class _UnpicklerWrapper(pickle._Unpickler):
    def __init__(self, importer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._importer = importer

    def find_class(self, module, name):
        # Subclasses may override this.
        if self.proto < 3 and self.fix_imports:
            if (module, name) in _compat_pickle.NAME_MAPPING:
                module, name = _compat_pickle.NAME_MAPPING[(module, name)]
            elif module in _compat_pickle.IMPORT_MAPPING:
                module = _compat_pickle.IMPORT_MAPPING[module]
        mod = self._importer.import_module(module)
        return getattr(mod, name)

def _get_all_files(filelist):
    # zip files don't always expictly list directories
    # so if we see foo/bar.py, add foo as a directory
    files = {} # name -> is_directory
    for file in filelist:
        # if we have already seen this file as part of a path
        # then it must be a directory, eventhough it is explicitly
        # written as an entry in the zip file. Otherwise,
        # it is a regular file.
        if file not in files:
            files[file] = 'regular'
        components = file.split('/')
        for i in range(1, len(components)):
            files['/'.join(components[:-i])] = 'directory'
    return files
