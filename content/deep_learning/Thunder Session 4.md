---
title: Quantization using Thunder Transforms
created: 2024-08-21 00:09
publish: true
tags:
  - DL
  - Compilers
path: /content/deep_learning/
---
# Background
## `llama.cpp`, `llm.c`
These are faster than PyTorch implementation. The reason is that, they have **minimal abstractions** on the kernels. In other words, it does not have different layers of different libraries which allows them to represent everything in a short stack. These characteristics provide a **finer control** over optimization (e.g. C frontend, CUDA kernels, how we load things to memory, etc). Using this, one can find the ideal optimization for a particular computation.
## Purpose of Thunder
- Finer control over optimization using PyTorch
Thunder strives to deliver the same advantages `llama.cpp` and `llm.c` have, using PyTorch. (Don't have to low-level) This means to provide the same level of control to optimize a given program.
- Avoid NxM problem
Different models can be optimized in different ways. This means that when we have N models M optimizations that can be applied to each model. If we can **separate** the **model** and the **optimization** tactic, this can be reduced to N + M (i.e. separate the 'what', what model/computation, from the 'how', as in how to make it fast).
- Make transforms a first class citizen

> You can't have a zoo for every llama. 
> - Thomas Viehmann
## Optimization
There can be many different ways to optimize a program. 
- Kernel
- Reducing memory pressure (i.e. quantization)
- Distributed
- etc
# Quantization
- Enables us to work with fewer bits
## Quantization in models
A model has a weight that is essentially a tensor that are represented with 32 bits or bfloat16 (half the size of 32 bits). 
## GPU
GPU has a main memory and multiple tensor cores. Consider the best case scenario where we were able to fuse all kernels into a single kernel. When calling that single kernel multiple times, we have to transfer data between the main memory and tensor cores several times (store and load data). That time `t` is dependent on the device's bandwidth. As the bandwidth is fixed, to speed this process, we can decrease the numerical precision of the data. If we reduce the precision to 4 bits, we can take 1/4 time compared to bfloat16. This is `t` is critical for large models weights ad KV cache.
## Bandwidth
Bandwidth is measured in GB/s. This is a **fixed** value for different devices. For instance, A10 has a bandwidth of 600 GB/s. This determines the speed of the data transformation between tensor cores and GPU's main memory. 

# Without Quantization
```python
import torch, thunder

with torch.device("cuda"):
	mlp = torch.nn.Sequential(
		torch.nn.Linear(512, 1024),
		torch.nn.GELU(),
		torch.nn.Linear(1024, 2),
	).requires_grad_(False).eval()
	inp = torch.randn(1, 512)

jm = thunder.jit(mlp)
thunder.last_traces(jm)[-1]
"""
# Constructed by Delete Last Used (took 0 milliseconds)
import torch
import torch.nn.functional
from thunder.executors.torchex import no_autocast

@torch.no_grad()
@no_autocast
def computation(input, t_0_bias, t_0_weight, t_2_bias, t_2_weight):
  # input: "cuda:0 f32[1, 512]"
  # t_0_bias: "cuda:0 f32[1024]"
  # t_0_weight: "cuda:0 f32[1024, 512]"
  # t_2_bias: "cuda:0 f32[2]"
  # t_2_weight: "cuda:0 f32[2, 1024]"
  t3 = torch.nn.functional.linear(input, t_0_weight, t_0_bias)  # t3: "cuda:0 f32[1, 1024]"
    # t3 = ltorch.linear(input, t_0_weight, t_0_bias)  # t3: "cuda:0 f32[1, 1024]"
      # t3 = prims.linear(input, t_0_weight, t_0_bias)  # t3: "cuda:0 f32[1, 1024]"
  del input, t_0_weight, t_0_bias
  t9 = torch.nn.functional.gelu(t3, approximate='none')  # t9: "cuda:0 f32[1, 1024]"
    # t9 = ltorch.gelu(t3, approximate='none')  # t9: "cuda:0 f32[1, 1024]"
      # t22 = ltorch.true_divide(t3, 1.4142135623730951)  # t22: "cuda:0 f32[1, 1024]"
        # t22 = prims.div(t3, 1.4142135623730951)  # t22: "cuda:0 f32[1, 1024]"
      # t23 = ltorch.erf(t22)  # t23: "cuda:0 f32[1, 1024]"
        # t23 = prims.erf(t22)  # t23: "cuda:0 f32[1, 1024]"
      # t24 = ltorch.mul(0.5, t23)  # t24: "cuda:0 f32[1, 1024]"
        # t24 = prims.mul(0.5, t23)  # t24: "cuda:0 f32[1, 1024]"
      # t25 = ltorch.add(0.5, t24, alpha=None)  # t25: "cuda:0 f32[1, 1024]"
        # t25 = prims.add(0.5, t24)  # t25: "cuda:0 f32[1, 1024]"
      # t9 = ltorch.mul(t3, t25)  # t9: "cuda:0 f32[1, 1024]"
        # t9 = prims.mul(t3, t25)  # t9: "cuda:0 f32[1, 1024]"
  del t3
  t13 = torch.nn.functional.linear(t9, t_2_weight, t_2_bias)  # t13: "cuda:0 f32[1, 2]"
    # t13 = ltorch.linear(t9, t_2_weight, t_2_bias)  # t13: "cuda:0 f32[1, 2]"
      # t13 = prims.linear(t9, t_2_weight, t_2_bias)  # t13: "cuda:0 f32[1, 2]"
  del t9, t_2_weight, t_2_bias
  return t13
"""

```
# With Quantization
```python
# quantization
# option 1. quantization can be a part of the model
# option 2. quantization can be viewed as an optimization -> this is what we are doing

from thunder.transforms.quantization import (
	BitsAndBytesLinearQuant4bit, 
	get_bitsandbytes_exeucutor,
)

bnb_executor = get_bitsandbytes_executor()

# apply quantization
jm = thunder.jit(
	mlp, 
	transforms=[BitsAndBytesLinearQuant4bit()], 
	executors=(bnb_executors)
)
out = jm(inp)
```
```python
thunder.last_traces(jm)[-1]
"""
# Constructed by Delete Last Used (took 0 milliseconds)
import torch
import torch.nn.functional
from thunder.executors.torchex import no_autocast

@torch.no_grad()
@no_autocast
def computation(input, t_0_bias, t_0_weight, t_2_bias, t_2_weight, t_0_weight_absmax, t_0_weight_code, t_2_weight_absmax, t_2_weight_code):
  # input: "cuda:0 f32[1, 512]"
  # t_0_bias: "cuda:0 f32[1024]"
  # t_0_weight: "cuda:0 ui8[262144, 1]"
  # t_2_bias: "cuda:0 f32[2]"
  # t_2_weight: "cuda:0 ui8[1024, 1]"
  # t_0_weight_absmax: "cuda:0 f32[8192]"
  # t_0_weight_code: "cuda:0 f32[16]"
  # t_2_weight_absmax: "cuda:0 f32[32]"
  # t_2_weight_code: "cuda:0 f32[16]"

  # /home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/linear.py:117: 	        return F.linear(input, self.weight, self.bias)
  t3 = bnb_matmul_nf4(input, t_0_weight, t_0_bias, t_0_weight_absmax, t_0_weight_code, 64, torch.float32, (1024, 512))  # t3: "cuda:0 f32[1, 1024]"
  del input, t_0_weight, t_0_bias, t_0_weight_absmax, t_0_weight_code
  t9 = torch.nn.functional.gelu(t3, approximate='none')  # t9: "cuda:0 f32[1, 1024]"
    # t9 = ltorch.gelu(t3, approximate='none')  # t9: "cuda:0 f32[1, 1024]"
      # t11 = ltorch.true_divide(t3, 1.4142135623730951)  # t11: "cuda:0 f32[1, 1024]"
        # t11 = prims.div(t3, 1.4142135623730951)  # t11: "cuda:0 f32[1, 1024]"
      # t12 = ltorch.erf(t11)  # t12: "cuda:0 f32[1, 1024]"
        # t12 = prims.erf(t11)  # t12: "cuda:0 f32[1, 1024]"
      # t14 = ltorch.mul(0.5, t12)  # t14: "cuda:0 f32[1, 1024]"
        # t14 = prims.mul(0.5, t12)  # t14: "cuda:0 f32[1, 1024]"
      # t15 = ltorch.add(0.5, t14, alpha=None)  # t15: "cuda:0 f32[1, 1024]"
        # t15 = prims.add(0.5, t14)  # t15: "cuda:0 f32[1, 1024]"
      # t9 = ltorch.mul(t3, t15)  # t9: "cuda:0 f32[1, 1024]"
        # t9 = prims.mul(t3, t15)  # t9: "cuda:0 f32[1, 1024]"
  del t3

  # /home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/linear.py:117: 	        return F.linear(input, self.weight, self.bias)
  t13 = bnb_matmul_nf4(t9, t_2_weight, t_2_bias, t_2_weight_absmax, t_2_weight_code, 64, torch.float32, (2, 1024))  # t13: "cuda:0 f32[1, 2]"
  del t9, t_2_weight, t_2_bias, t_2_weight_absmax, t_2_weight_code
  return t13
"""
```
The steps have occurred in the above code:
1. quantized the weights
2. get all quantized weights to the computation
3. In the computation, replace the usual matrix multiplication or linear layer to the quantized ones (i.e. `linear` layers are now using `bnb_matmul_nf4`)
Notice that the computation has **more arguments** which are needed for quantization. The added arguments are the following for this particular example: `t_0_weight_absmax, t_0_weight_code, t_2_weight_absmax, t_2_weight_code`. These new tensors are used to scale the tensors when performing quantization.
# Advantages of Thunder Transformation
1. We can easily **compose** different Transformations (e.g. quantization, distributed, etc)
Can have different transformations and executors for different operations. By adding the default executor which is NVFuser, we get our fusion back.
```python
jm = thunder.jit(
	mlp, 
	transforms=[BitsAndBytesLinearQuant4bit(), thunder.distributed.fsdp()], 
	executors=(bnb_executors, *thunder.get_default_executors(), )				 
)
```
2. We can do all the manipulation **without** **allocating** any tensor. This is possible because all acquisition relies on the meta data of the tensor (i.e. shape and type of the tensor, not the values in the tensor) We can do quantization first before materializing the model on the GPU.
# Reference
- https://youtu.be/t9Fj5VjIpac?feature=shared