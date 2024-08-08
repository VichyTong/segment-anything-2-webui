# Segement Anything 2 WebUI

Please follow the instruction to install the SAM 2 first.
> SAM 2 needs to be installed first before use. The code requires `python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. You can install SAM 2 on a GPU machine using:

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git

cd segment-anything-2; pip install -e .
```

Then, install the requirements for the webui:

```bash
pip install -r requirements.txt
```

To download the checkpoints, we need to run the following command:

```bash
cd checkpoints
./download_ckpts.sh
```


To run the webui, use the following command:

```bash
python app.py
```