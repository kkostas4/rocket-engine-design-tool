"""conftest.py — project-root pytest configuration.

Adds the repo root to sys.path so that ``import physics`` resolves
without installing the package.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
