# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path

import pytest


@pytest.fixture()
def chignolin_pdb():
    return Path(__file__).parent / "chignolin.pdb"
