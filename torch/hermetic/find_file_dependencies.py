import types
from typing import List, Dict, Tuple, Set, Union, Optional
import ast
import sys
from ._importlib import _resolve_name

class _ExtractModuleReferences(ast.NodeVisitor):
    """
    Extract the list of global variables a block of code will read and write
    """

    @classmethod
    def run(cls, src: str, package: str) -> List[str]:
        visitor = cls(package)
        tree = ast.parse(src)
        visitor.visit(tree)
        return list(visitor.references.keys())

    def __init__(self, package):
        super().__init__()
        self.package = package
        self.references = {}

    def _absmodule(self, module_name: str, level: int) -> str:
        if level > 0:
            return _resolve_name(module_name, self.package, level)
        return module_name

    def visit_Import(self, node):
        for alias in node.names:
            self.references[alias.name] = True

    def visit_ImportFrom(self, node):
        name = self._absmodule(node.module, 0 if node.level is None else node.level)
        self.references[name] = True

find_files_source_depends_on = _ExtractModuleReferences.run
