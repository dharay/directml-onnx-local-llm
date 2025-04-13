---
license: mit
pipeline_tag: text-generation
tags:
 - ONNX
 - DML
 - ONNXRuntime
 - phi3
 - nlp
 - conversational
 - custom_code
inference: false
---

# Phi-3 Medium-4K-Instruct ONNX DirectML models

<!-- Provide a quick summary of what the model is/does. -->
This repository hosts the optimized versions of [Phi-3-medium-4k-instruct](https://aka.ms/phi3-medium-4K-instruct) to accelerate inference with DirectML and ONNX Runtime for your machines with GPUs. 

Phi-3 Medium is a 14B parameter, lightweight, state-of-the-art open model trained with the Phi-3 datasets, which include both synthetic data and the filtered publicly available websites data, with a focus on high-quality and reasoning dense properties. The model belongs to the Phi-3 family with the medium version in two variants: [4K](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) and [128K](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct), which are the context lengths (in tokens) that they can support.

The model has underwent a post-training process that incorporates both supervised fine-tuning and direct preference optimization for the instruction following and safety measures. When assessed against benchmarks testing common sense, language understanding, math, code, long context, and logical reasoning, Phi-3-Medium-4K-Instruct showcased a robust and state-of-the-art performance among models of the same-size and next-size-up.

Optimized variants of the Phi-3 Medium models are published here in [ONNX](https://onnx.ai) format and run with [DirectML](https://learn.microsoft.com/en-us/windows/ai/directml/dml-intro). This lets developers bring hardware acceleration to Windows devices at scale across AMD, Intel, and NVIDIA GPUs. 

## ONNX Models

Here are some of the optimized configurations we have added:

1. ONNX model for INT4 DML: ONNX model optimized to run with DirectML and quantized to int4 precision using AWQ*.

How do you know which is the best ONNX model for you:
- Are you on a Windows machine with GPU?
    - I don't know → Review this [guide](https://www.microsoft.com/en-us/windows/learning-center/how-to-check-gpu) to see whether you have a GPU in your Windows machine.
    - Yes → Access the Hugging Face DirectML ONNX models and instructions at [Phi-3-medium-4k-instruct-onnx-directml](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct-onnx-directml).
    - No → Do you have a NVIDIA GPU?
        - I don't know → Review this [guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#verify-you-have-a-cuda-capable-gpu) to see whether you have a CUDA-capable GPU.
        - Yes → Access the Hugging Face CUDA ONNX models and instructions at [Phi-3-medium-4k-instruct-onnx-cuda](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct-onnx-cuda) for NVIDIA GPUs.
        - No → Access the Hugging Face ONNX models for CPU devices and instructions at [Phi-3-medium-4k-instruct-onnx-cpu](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct-onnx-cpu)

## How to Get Started with the Model
To support the Phi-3 models across a range of devices, platforms, and EP backends, we introduce a new API to wrap several aspects of generative AI inferencing. This API makes it easy to drag and drop LLMs straight into your app. To run the early version of these models with ONNX, follow the steps [here](http://aka.ms/generate-tutorial). You can also test this with a [chat app](https://github.com/microsoft/onnxruntime-genai/tree/main/examples/chat_app).

## Hardware Supported

The model has been tested on:
- GPU SKU: RTX 4090 (DirectML)

Minimum Configuration Required:
- Windows: DirectX 12-capable GPU and a minimum of 10GB of combined RAM

### Model Description

- **Developed by:**  Microsoft
- **Model type:** ONNX
- **Language(s) (NLP):** Python, C, C++
- **License:** MIT
- **Model Description:** This is a conversion of the Phi-3 Medium-4K-Instruct model for ONNX Runtime inference.

## Additional Details
- [**Phi-3 Small, Medium, and Vision Blog**](https://aka.ms/phi3_ONNXBuild24) and [**Phi-3 Mini Blog**](https://aka.ms/phi3-optimizations)
- [**Phi-3 Model Blog Link**](https://aka.ms/phi3blog-april)
- [**Phi-3 Model Card**]( https://aka.ms/phi3-medium-4k-instruct)
- [**Phi-3 Technical Report**](https://aka.ms/phi3-tech-report)
- [**Phi-3 on Azure AI Studio**](https://aka.ms/phi3-azure-ai)

## Performance Metrics

## DirectML
We measured the performance of DirectML and ONNX Runtime's new Generate() API with Phi-3 medium quantized with Activation-Aware Quantization [AWQ](https://arxiv.org/abs/2306.00978) and with a block size of 128 on Windows. Our test machine had an NVIDIA GeForce RTX 4090 GPU and an Intel Core i9-13900K CPU. DirectML lets developers not only achieve great performance but also lets developers deploy models across the entire Windows ecosystem with support from AMD, Intel, and NVIDIA. Best of all, AWQ means that developers get this scale while also maintaining high model accuracy.

Stay tuned for additional performance improvements in the coming weeks thanks to optimized drivers from our hardware partners, along with additional updates to the ONNX Runtime Generate() API.

| Batch Size, Prompt Length | Block Size = 32 |	Block Size = 128 |
|---------------------------|-----------------|------------------|	
| 1, 16 | 66.36 | 72.39 |


#### Package Versions

| Pip package name | Version |
|------------------|---------|
| torch            | 2.2.0   |
| triton           | 2.2.0   |
| onnxruntime-gpu  | 1.18.0  |
| transformers     | 4.39.0  |
| bitsandbytes     | 0.42.0  |

## Appendix

### Activation Aware Quantization
AWQ works by identifying the top 1% most salient weights that are most important for maintaining accuracy and quantizing the remaining 99% of weights. This leads to less accuracy loss from quantization compared to many other quantization techniques. For more on AWQ see [here](https://arxiv.org/abs/2306.00978).

## Model Card Contact
parinitarahi, kvaishnavi, natke

## Contributors
Kunal Vaishnavi, Sunghoon Choi, Yufeng Li, Sheetal Arun Kadam, Natalie Kershaw, Parinita Rahi, Patrice Vignola, Xiang Zhang, Chai Chaoweeraprasit, Logan Iyer, Vicente Rivera, Jacques Van Rhyn
