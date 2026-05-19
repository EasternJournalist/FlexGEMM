import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_cuda = pytest.mark.skip(reason="CUDA is required")
    for item in items:
        item.add_marker(skip_cuda)
