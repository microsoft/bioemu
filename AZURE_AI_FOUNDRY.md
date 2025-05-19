# 🚀 How to run BioEmu on Azure AI Foundry

[![](https://dcbadge.limes.pink/api/server/https://discord.gg/azureaifoundry)](https://discord.gg/azureaifoundry)

First, navigate to [BioEmu](https://ai.azure.com/explore/models/BioEmu/version/1/registry/azureml) in the Azure AI Foundry [model catalog](https://ai.azure.com/explore/models).

Then click on "Sign in" button (if you don't have an Azure account, you can create one for free).

Once you are signed-in, you will see a "Deploy" button. Click on it. Choose a project (create a new one if you don't have it already). Select one instance of "Standard_NC24ads_A100_v4" virtual machine (the default) and click "Deploy". The price may depend on the region you choose to create the deployment.

After a while (expect 30 minutes), when the endpoint and deployment are ready, you will see a "Consume" tab next to "Details". Copy the endpoint URL and API key.

When the region you have selected does not have enough capacity, the deployment may get stuck in "Creating" state due to retries. Try a different region in this case.

Prepare a python environment with [`requests` package](https://pypi.org/project/requests/), which can be installed as `pip install requests`.

To send a request and decode the response from an online deployment, run the following code:
```python
import base64
import os
import sys

import requests

ENDPOINT_URL = "https://<ONLINE_ENDPOINT_NAME>.<REGION>.inference.ml.azure.com/score"
ENDPOINT_API_KEY = "<ONLINE_ENDPOINT_API_KEY>"
OUTPUT_DIR = os.path.expanduser("~/test-chignolin")

SEQUENCE = "GYDPETGTWG"
NUM_SAMPLES = 10

def base64_decode_results(output_dir: str, result: dict[str, str]) -> None:
    """
    Decode the results from base64 format.
    """
    os.makedirs(output_dir, exist_ok=True)
    for file_name, raw_data in result.items():
        with open(os.path.join(output_dir, file_name), "wb") as f:
            f.write(base64.b64decode(raw_data.encode("utf-8")))


headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": ("Bearer " + ENDPOINT_API_KEY),
}
data = {
    "input_data": {
        "sequence": SEQUENCE,
        "num_samples": NUM_SAMPLES,
    }
}

response = requests.post(url=ENDPOINT_URL, headers=headers, json=data)
try:
    result = response.json()
except BaseException as e:
    raise RuntimeError(
        f"Got status code {response.status_code}. Detailed message\n{response.text}"
    ) from e
if result["status"] != "success":
    raise RuntimeError(f"Inference failed with the following error:\n{result['message']}")

print(f"Writing results to {OUTPUT_DIR}")
base64_decode_results(output_dir=OUTPUT_DIR, result=result["results"])
```

`samples.xtc`, `sequence.fasta`, `topology.pdb` files are written to `OUTPUT_DIR`. Please see the main [`README.md`](README.md) for instructions on further analysis.


> [!TIP]
> Sign-in to Azure is required if you wish to browse and download the model files (checkpoint and config) from the "Artifact" tab.

Please join the [discord channel](https://discord.gg/azureaifoundry) if you have general questions about Azure AI Foundry.