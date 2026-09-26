"""Shared pytest config: `--gpu` opt-in for GPU-marked tests; repo root as cwd (the eval
gates address caches by repo-relative path, as the pipeline itself does)."""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", help="also run tests marked gpu")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU; runs only with --gpu")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpu"):
        return
    skip = pytest.mark.skip(reason="GPU test: pass --gpu to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def _repo_cwd():
    old = os.getcwd()
    os.chdir(REPO_ROOT)
    yield
    os.chdir(old)
