# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path

import numpy as np
import torch


def test_digconditional_score_model(default_batch, tiny_model):
    model = tiny_model
    model.eval()
    model_output = model(default_batch, t=torch.tensor([0.0] * 10))
    model_output = {
        k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        for k, v in model_output.items()
    }
    expected_path = Path(__file__).parent / "expected.npz"
    # Uncomment below to update expected model output.
    # with open(expected_path, "wb") as f:
    #     np.savez(f, **model_output)
    expected_output = np.load(expected_path)
    assert set(model_output.keys()) == set(expected_output.keys())
    for k in model_output.keys():
        if k == "system_id":
            assert model_output[k] == expected_output[k].tolist()
        else:
            assert np.allclose(model_output[k], expected_output[k], atol=1e-5)
