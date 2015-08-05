#!/usr/bin/julia

using CUDArt

result = devices(dev->true) do devlist
    d_A = CudaArray(Float64, (200,300))
    d_B = CudaPitchedArray(Int32, (15,40,27))
    
    h_A = HostArray(Float32, (1000,1200))


end

