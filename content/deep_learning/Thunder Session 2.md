---
title: Why Deep Learning Compilers
created: 2024-08-20 02:04
publish: true
tags:
  - DL
  - Compilers
path: /content/deep_learning/
---
# GPU
- GPUs does two things:
	- **Computes** operations
	- **Load** things from memory and **stores** them back
# Problems with GPUs
- CPU **queue** kernels on the GPU
- GPU **asynchronously** executes the kernels
	- Some operations are fast (e.g. ReLU)
	- Some are slow (e.g. Linear)
> [!faq] Can the CPUs actually feed the GPUs **fast** enough?
- GPU wastes its time doing things that are not computing
	- Computations are done in the tensor cores (which has gotten a lot faster)
	- **Between** kernels, we always **write back** to the global GPU memory
> [!faq] Can we reduce the amount of memory transfer? (i.e. we want to spend as much time as possible in the tensor cores)
# Kernel
- One function that we execute in the GPU
- Self-contained
	- **Everything** that's coming into the kernel comes from memory
	- All the returned result has to be written to the memory
- Kernel launch time
# How do we combine multiple kernels to one big kernel?
## Option 1. Write a CUDA code directly
This essentially means that we are writing the code from scratch. A famous example could be [llm.c](https://github.com/karpathy/llm.c).
## Option 2. Use something that can convert my PyTorch code to something that is fast
### Why do we want to use PyTorch?
- It allows us to iterate quickly on ideas (fast to implement compared to writing C or CUDA program)
### How do we make the PyTorch code fast?
- Examples of "tools" (compilers)
	- Apache TVM, `torch.compile`, torch script, JAX
# How does the compiler work?
- PyTorch Function -> Intermediate representation -> GPU code / CUDA code / LLVM representation to generate PTX
- Detailed explanations are provided in [[Thunder Session 3]]
# Visualizing the difference between PyTorch eager, `torch.compile`, and Thunder
## Common code
```python
def print_device_time(prof):
	for k in sorted(prof.key_average(), key=lambda k: k.device_time, reverse=True):
		print(f"{k.device_time:.3f}us       {k.key}")
import torch, thunder
import math
def my_gelu(x):
	return 0.5 * x * (1 + torch.tanh((2 / math.pi)**0.5 (a + 0.4475 * x **3)))

x = torch.randn(5, 5, device="cuda")
```
## PyTorch eager
```python
import dis # python disassembly module
dis.dis(my_gelu) # decomposes the operations into many steps

with torch.profiler.profile() as prof:
	my_gelu(x)
print_device_time(prof) 
# shows how much time is spent on each step in the GPU
# prints all kernels and its time it took to complete the computation
```
When PyTorch executes, it has a lot of mechanisms to dispatch to the right kernel. It will select the best kernel to run in that particular situation. (e.g. can have multiple kernels for one operation)
## `torch.compile`
```python
my_gelu_compiled = torch.compile(my_gelu)
with torch.profiler.profile() as prof:
	my_gelu_compiled(x)

print_device_time(prof)
# it shows that torch.compile uses triton as its backend
# torch.compile generates triton code
# that triton code is compiled to the GPU (it can generate kernels)
```
## Thunder
```python
my_gelu_jit = thunder.jit(my_gelu)
my_gelu_jit(x)

with torch.profiler.profile() as prof:
	my_gelu_jit(x)
print_device_time(prof)
# it shows that we use NVFuser kernel (backend)

thunder.last_traces(my_gelu_jit)[-1]
```

## Observations
- The speed of computing a single kernel is **similar** to computing one big kernel
- We are **less** loading from memory and saving it back
- i.e. spending more time computing

# Reference
- https://youtu.be/Od1STXifgjE?feature=shared


