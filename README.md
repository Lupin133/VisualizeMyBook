# Visualize My Book

Visualize My Book is an innovative project that leverages Large Language Models (LLM) and advanced image generation technology (SDXL) to empower anyone to illustrate their stories quickly and effortlessly using AI. This tool is invaluable for authors and readers alike who are looking to enrich their narratives with visual support.

# Full Installation Guide  

This guide will walk you through the process of setting up the environment and installing the necessary components to run the Visualize My Book project.  
⚠️ Video for complete installation guide : https://gemoo.com/tools/upload-video/share/619011923937320960?codeId=MpKnekgNAeA3N&card=619011920913227776

## Environment Setup
You will need to have TensorRT and CUDA 12 installed on your system. It is recommended to use Conda to create a new Python 3.10 environment:

```bash
conda create -n VisualizeMyBook python=3.10
```

Activate the new environment:

```bash
conda activate VisualizeMyBook
```

## LLM Engine Setup
Follow the TensorRT installation guide, clone the TensorRT-LLM repository (release branch) and navigate to the examples/llama folder.

Download the Mistral model by running the following command:

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/mistral-7b-int4-chat/versions/1.1/zip -O mistral-7b-int4-chat_1.1.zip
```
If you are unable to use wget, you can download the model from the link provided in the command and rename the downloaded file to mistral-7b-int4-chat_1.1.zip.

Extract the folder, rename it `mistral_awq` and the final structure should look like:

```shell
mistral_awq/
├── mistral7b_hf_tokenizer
├── mistral_kv_int8_scales
├── mistral_tp1.json
├── mistral_tp1_rank0.npz
```

Run the following command to build the engine:
```bash
python build.py --model_dir mistral_awq/mistral7b_hf_tokenizer --quant_ckpt_path mistral_awq/mistral_tp1_rank0.npz --dtype float16 --remove_input_padding --use_gpt_attention_plugin float16 --enable_context_fmha --use_gemm_plugin float16 --use_weight_only --weight_only_precision int4_awq --per_group --output_dir converted --int8_kv_cache --ft_model_dir mistral_awq/mistral_kv_int8_scales --max_input_len 32256 --max_batch_size 1
```

Put the `mistral_awq` folder in the root of the project

## SDXL Engine Setup

It's not necessary, depending on your setup, but I recommend installing nvtx and mpi4py to avoid any errors :

```bash
pip install https://huggingface.co/Superd4/test/resolve/main/nvtx-0.2.10-cp310-cp310-win_amd64.whl
conda install -c intel mpi4py
```

Install the required packages using pip:
```bash
pip install -r requirements.txt
```

Run the following command to download the necessary checkpoint:

```bash
python download_ckpt.py
```

## Running

```bash
python gradio_gui.py
```

# Source

I use [this repo](https://github.com/phineas-pta/SDXL-trt-win) as a base for tensorRT with SDXL on Windows natively without docker.
I took the first 2 chapters of a [random wattpad fiction](https://www.wattpad.com/50799759-the-other-ceo-chapter-1) for the pdf.