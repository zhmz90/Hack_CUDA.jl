#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_REPS 100
#define TILE_DIM 16


texture<float, 2, cudaReadModeElementType> texRefPL;
texture<float, 2, cudaReadModeElementType> texRefArray;


__global__ void shift_pitch_linear(float *odata, int pitch, int width, int height, int shiftX, int shiftY)
{
    int gtx = blockIdx.x * blockDim.x + threadIdx.x;
    int gty = blockIdx.y * blockDim.y + threadIdx.y;

    odata[gty*pitch+gtx] = tex2D(texRefPL, (gtx+shiftX)/(float)width, (gty+shiftY)/(float)height);
}

__global__ void shift_array(float *odata, int pitch, int width, int height, int shiftX, int shiftY)
{
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = blockIdx.y * blockDim.y + threadIdx.y;

    odata[yid*pitch + xid] = tex2D(texRefArray, (xid+shiftX)/(float)width, (yid+shiftY)/(float)height);

}

void run_test(int argc, char **argv)
{
    const int nx = 2048;
    const int ny = 2048;

    const int x_shift = 5;
    const int y_shift = 7;

    dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM), dimBlock(TILE_DIM, TILE_DIM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float *h_idata = (float *)malloc(sizeof(float)*nx*ny);
    float *h_odata = (float *)malloc(sizeof(float)*nx*ny);
    float *gold = (float *)malloc(sizeof(float)*nx*ny);

    for(int i=0; i< nx*ny;i++)
    {
        h_idata[i] = (float)i;
    }
    
    float *d_idataPL;
    size_t d_pitchBytes;
    cudaMallocPitch((void **)&d_idataPL, &d_pitchBytes, nx*sizeof(float), ny);

    cudaArray *d_idataArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaMallocArray(&d_idataArray, &channelDesc, nx, ny);

    float *d_odata;
    cudaMallocPitch(&d_odata, &d_pitchBytes, nx*sizeof(float), ny);

    size_t h_pitchBytes = nx*sizeof(float);
    cudaMemcpy2D(d_idataPL, d_pitchBytes, h_idata, h_pitchBytes, nx*sizeof(float),ny,
            cudaMemcpyHostToDevice);

    cudaMemcpyToArray(d_idataArray,0,0,h_idata,nx*ny*sizeof(float),
            cudaMemcpyHostToDevice);

    texRefPL.normalized = 1;
    texRefPL.filterMode = cudaFilterModePoint;
    texRefPL.addressMode[0] = cudaAddressModeWrap;
    texRefPL.addressMode[1] = cudaAddressModeWrap;

    cudaBindTexture2D(0, &texRefPL, d_idataPL, &channelDesc, nx, ny, d_pitchBytes);

    texRefArray.normalized = 1;
    texRefArray.filterMode = cudaFilterModePoint;
    texRefArray.addressMode[0] = cudaAddressModeWrap;
    texRefArray.addressMode[1] = cudaAddressModeWrap;

    cudaBindTextureToArray(texRefArray, d_idataArray, channelDesc);


    for (int j=0; j<ny;++j)
    {
        int jshift = (j+y_shift) % ny;

        for (int i=0;i<nx;i++)
        {
            int ishift = (i+x_shift) % nx;
            gold[j*nx+i] = h_idata[jshift*nx+ishift];
        }
    }

    cudaMemset2D(d_odata, d_pitchBytes, 0, nx*sizeof(float),ny);
    cudaEventRecord(start, 0);

    for(int i=0; i<NUM_REPS;i++)
    {   
        shifPitchLinear<<<dimGrid,dimBlock>>>(d_odata,(int)(d_pitchBytes/sizeof(float)),
                nx,ny,x_shift,y_shift);
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float timePL;
    cudaEventElapsedTime(&timePL, start, stop);

    cudaMemcpy2D(h_odata, h_pitchBytes, d_odata, d_pitchBytes, nx*sizeof(float), ny,
            cudaMemcpyDeviceToHost);

    // shiftArray kernel
    cudaMemset2D(d_odata, d_pitchBytes, 0, nx*sizeof(float), ny);
    cudaEventRecord(start,0);

    for (int i=0; i<NUM_REPS;i++)
    {
        shiftArray<<<dimGrid,dimBlock>>>(d_odata, (int)(d_pitchBytes/sizeof(float)),
                nx,ny,x_shift,y_shift);
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float timeArray;
    cudaEventElapsedTime(&timeArray, start, stop);

    cudaMemcpy2D(h_odata,h_pitchBytes, d_odata, d_pitchBytes,nx*sizeof(float),ny,
            cudaMemcpyDeviceToHost);

    free(h_idata);
    free(h_odata);
    
    cudaUnbindTexture(texRefPL);
    cudaUnbindTexture(texRefArray);

    cudaFree(d_idataPL);
    cudaFreeArray(d_idataArray);
    cudaFree(d_odata);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

int main(int argc, char **argv)
{   
    run_test(argc, argv);

    cudaDeviceReset();
    return 0;
}
