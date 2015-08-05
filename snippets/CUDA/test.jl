#!/usr/bin/julia

using CUDA

dev = CuDevice(0)

ctx = create_context(dev)

md = CuModule("vadd.ptx")

vadd = CuFunction(md, "vadd")

a = round(rand(Float32, (3,4))*100)
b = round(rand(Float32, (3,4))*100)
ga = CuArray(a)
gb = CuArray(b)

gc = CuArray(Float32, (3,4))
launch(vadd,12,1,(ga,gb,gc))

c = to_host(gc)

free(ga)
free(gb)
free(gc)

@show a
@show b
@show c
@show c == a+b
unload(md)
destroy(ctx)

