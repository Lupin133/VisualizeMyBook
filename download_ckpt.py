# -*- coding: utf-8 -*-

from huggingface_hub import snapshot_download

snapshot_download(
	repo_id="stabilityai/stable-diffusion-xl-1.0-tensorrt",
	local_dir="onnx-ckpt",
	local_dir_use_symlinks=False,
	resume_download=True
)
