"""Compatibility import; new users should import sgl_analytic.subhalos_exact."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("sgl_analytic.subhalos_exact")
