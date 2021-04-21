/*
Copyright (c) 2021 Alexandre Sirois-Vigneux

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

   1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.

   3. This notice may not be removed or altered from any source
   distribution.
*/


#ifndef VOLUME_H
#define VOLUME_H

#include "cuda_runtime.h"
#include "utils/helper_cuda.h"
#include "utils/helper_math.h"


class Volume {

public:
    cudaArray             *content;
    cudaExtent            size;
    cudaChannelFormatDesc channelDesc;
    cudaTextureObject_t   volumeTex;
    cudaSurfaceObject_t   volumeSurf;

    Volume(int w, int h, int d){
        // create 3D array
        cudaExtent dataSize = {(size_t)w, (size_t)h, (size_t)d};
        channelDesc = cudaCreateChannelDesc<float>();
        checkCudaErrors(cudaMalloc3DArray(&content, &channelDesc, dataSize, cudaArraySurfaceLoadStore));
        size = dataSize;

        cudaResourceDesc surfRes;
        memset(&surfRes, 0, sizeof(cudaResourceDesc));
        surfRes.resType = cudaResourceTypeArray;
        surfRes.res.array.array = content;

        checkCudaErrors(cudaCreateSurfaceObject(&volumeSurf, &surfRes));

        cudaResourceDesc texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));

        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array = content;

        cudaTextureDesc texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));

        texDescr.filterMode     = cudaFilterModeLinear;
        texDescr.addressMode[0] = cudaAddressModeClamp;
        texDescr.addressMode[1] = cudaAddressModeClamp;
        texDescr.addressMode[2] = cudaAddressModeClamp;

        checkCudaErrors(cudaCreateTextureObject(&volumeTex, &texRes, &texDescr, NULL));
    }

    ~Volume() {
        checkCudaErrors(cudaDestroyTextureObject(volumeTex));
        checkCudaErrors(cudaDestroySurfaceObject(volumeSurf));
        checkCudaErrors(cudaFreeArray(content));
        content = 0;
    }
};


#endif
