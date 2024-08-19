---
created: <% tp.file.creation_date() %>
tags:
  - DL
  - Compilers
title: Introduction to Deep Learning Compilers
publish: true
path: /content/deep_learning/
---
# Deep Learning Compilers
When referring to deep learning compilers, there can be different levels of compilers that one is referring to. There are three steps that one can take to **run the code**.
1. Take the user's code as an input and **generate intermediate representations** (often referred as a **frontend** or **acquisition**)
2. Transform the representations if needed (e.g. optimization, quantization, fusing operations, etc)
3. Execute the code (often referred as a **backend** or **execution**)

> [!tip] Deep Learning Compiler == How do I get my code to run on the GPU?
## Examples of Deep Learning Compilers
- Thunder
	- Generates **description of the computation** and transforms it (fyi, these descriptions are called `trace`)
- Torch Compile
- JAX
- Triton
	- Turn a specification into a **kernel**
# Workflow of Thunder
1. User PyTorch code
2. Use `thunder.jit` to create a **jitted_fn**
3. When calling the **jitted_fn**, it generates `traces` (i.e. **acquisition**)
4. Transforms are done with the `traces`
5. Perform **transform for execution** (i.e. backend, execution)
## What happens when we do `thunder.jit`?
- Wrap the function that is going to run instead of the original function itself
- Cheap operation
## When running the jitted function, what happens?
- Thunder's python interpreter will run the function
	- It will replace
		- Tensor to **TensorProxy**
		- Calls to PyTorch to calls to **Thunder**
- Produces a representation of the program called **trace**
### How is it different from running a PyTorch Function (PyTorch Eager)?
- When we run a PyTorch function with an input argument, the following happens ...
	- Execute kernel
	- Allocate memory
	- Produce result
> [!tip] PyTorch eager computes the function directly while Thunder's jitted function generates a trace that represents the computation, optimize it, then performs the computation.
### Acquiring compile data
```python
cd = thunder.compile_data(jfn)
cd.get_computation_and_inputs(x) # get computation and inputs (produce all traces)
```
## What is `trace`?
- Thunder's **intermediate representation**
- Representation of the computation **before** the computation actually happens
	 - It produces an equivalent computation that will produce an equivalent results for equivalents inputs
- Represents what happens to inputs
	- Traces the meta data of the data
		- e.g. input tensor has a shape `10x10` and the output tensor has a shape of `20x20`
	- Treat inputs symbolically
- Does **not** allocate anything
- Contains computation Thunder wants to execute (series of torch functions)
### View traces
```python
lt = thunder.last_traces(jfn) # list of traces of the last execution
thunder.last_traces(jfn)[0] # original program (pure representation; no fusion)
thunder.last_traces(jfn)[-1] # transform for execution (also shows the backend)
```
## What kind of transforms can we do with the traces?
- Distributed
- Different numerical precision
- Can break down computations into different pieces
- Can fuse different operations
- Quantization
> [!tip] A cool thing that we can do is, with these transformations, we can quantize and fit a tensor that was too big before to the desired device!
## How do we execute it?
- Performs **transform for execution**
- Decides which bit should be handled with which backend
	- e.g. NVFuser
- We can leverage specialized libraries like fast attention, cuDNN, fuse different operations, etc
## Execution
- take the trace and executed
- tensors are allocated on the GPU
- kernels are launched
### What does the NVFuser do?
```python
fdw = lt.python_ctx()["nvFusion0"] # access everything that we need to know to execute
fdw.last_used # shows the nvFuser program
fdw.last_used.cuda_code_for([x]) # this returns a CUDA kernel that nvFuser produces
```
# Torch Compile
- Front end is called **Dynamo**
	- Dynamo produces **FxGraph** which is an intermediate representation
- FxGraph is handed off to the backend, typically, the **Inductor**
- Inductor generates code for **Triton**
- Triton will use be used to compile a **PTX kernel** 
	- PTX is platform independent across various GPUs from NVIDIA
- PTX kernel is translated to a **GPU kernel** and that will run on the GPU
### How does Dynamo produce FxGraph?
```python
def my_backend(gm, inps):
	gm.graph.print_tabular() # tabular representation
	gm.print_readable() # readable representation
	return gm.forward

torch._dynamo.reset()

cfn = torch.compile(my_fn, backend=my_backend)
cfn(x) # shows the computation and it computes the function
```
### How does the Inductor execute the code?
```python
import os
os.environ('TORCH_COMPILE_DEBUG') = '1'
import torch

cfn(x) 
# this will return a result and produce multiple files that shows debug information from the inductor
# e.g. output_code.py shows the output of the inductor (i.e. Triton kernel)
```
# JAX
- In between a framework and a compiler that is known for being **fast**
- it allows you to express computations that you can transform
- JAX **starts** from the intermediate representation
- `jaxpr` : jax expression (something like a intermediate representation)
- this generates the **HLO** operations
- XLA: kernel generating library
	- use LLVM to generate PTX (code on the GPU)
		- LLVM is a system to build compilers (powers clang)
### How do generate HLO operations in JAX?
```python
import os
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/jaxdebug'
import jax, numpy
import math

def jax_gelu(x):
	return 0.5 * x * (1 jax.numpy.tanh((2 / math.i)**0.5 * (x + 0.4475 * x ** 3)))

x = numpy.ones(16)
jax.make_jaxpr(jax_gelu)(x) # returns jax's intermediate representation
lowering = jax.jit(jax_gelu).lower(x) # lower level (HLO) representation
compiled = lowering.compile()
print(compiled.as_text()) # shows the HLO modules
```
### How do we get the LLVM representation?
After compiling in JAX, we have `.ll` files that are the LLVM representations.
# Thunder vs Torch Compile vs JAX

|                               | Thunder | Torch compile | JAX         |
| ----------------------------- | ------- | ------------- | ----------- |
| Can accept PyTorch functions? | Yes     | Yes           | No          |
| Intermediate Representation   | `trace` | `FxGraph`     | `jaxpr`<br> |
| (typical) Backend             | NVFuser | Inductor      | XLA         |
- Accepting PyTorch functions mean that the compiler has a **frontend** that can convert the PyTorch function to intermediate representations
- Backend is responsible for executing the code
# Reference
- https://youtu.be/HtL-T1nw0Rw?feature=shared



