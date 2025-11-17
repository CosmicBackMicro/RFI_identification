#!/usr/bin/env python3
import time
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

E='model_segformerB2_int8.engine'
f=open(E,'rb')
rt=trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine=rt.deserialize_cuda_engine(f.read())
f.close()
nb=engine.get_nb_bindings()
ctx=engine.create_execution_context()
in_idx=0
for i in range(nb):
    if engine.binding_is_input(i):
        in_idx=i
        break
shape=tuple(ctx.get_binding_shape(in_idx))
if any(d<=0 for d in shape):
    shape=(1,1,512,512)
inp=np.random.rand(*shape).astype(trt.nptype(engine.get_binding_dtype(in_idx)))
bindings=[0]*nb
dptr=[None]*nb

for i in range(nb):
    dt=trt.nptype(engine.get_binding_dtype(i))
    bshape=tuple(ctx.get_binding_shape(i))
    host=np.ascontiguousarray(inp.reshape(bshape).astype(dt)) if engine.binding_is_input(i) else np.empty(bshape,dtype=dt)
    d=cuda.mem_alloc(host.nbytes)
    bindings[i]=int(d)
    dptr[i]=d
    if engine.binding_is_input(i):
        cuda.memcpy_htod(bindings[i], host.tobytes())
start=time.perf_counter()
ctx.execute_v2(bindings)
lat=time.perf_counter()-start
for i in range(nb):
    if not engine.binding_is_input(i):
        name=engine.get_binding_name(i)
        dt=trt.nptype(engine.get_binding_dtype(i))
        bshape=tuple(ctx.get_binding_shape(i))
        out=np.empty(bshape,dtype=dt)
        cuda.memcpy_dtoh(out,dptr[i])
        print(name,out.shape,out.dtype)
print('latency',lat)
 