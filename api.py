"""Thin re-export so `uvicorn api:app --reload` works (Python can't import
modules whose names start with a digit, like 06_api.py)."""
import importlib
_mod = importlib.import_module("06_api")
app = _mod.app
state = _mod.state
