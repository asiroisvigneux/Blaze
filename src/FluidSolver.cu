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


#include "../thirdparty/cuda-noise/cuda_noise.cuh"

#include "FluidSolver.h"


__global__ void create_grids_kernel(Grid **d_mT, Grid **d_mU, Grid **d_mV, Grid **d_mW,
                             float *d_mTFront, float *d_mTBack, float *d_mUFront, float *d_mUBack,
                             float *d_mVFront, float *d_mVBack, float *d_mWFront, float *d_mWBack,
                             int width, int height, int depth, float dx, SceneSettings scn)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_mT = new Grid(d_mTFront, d_mTBack, width, height, depth, 0.5f, 0.5f, 0.5f, dx,
                         scn.domainBboxMin, scn.domainBboxMax, false, scn);
        *d_mU = new Grid(d_mUFront, d_mUBack, width+1, height, depth, 0.0f, 0.5f, 0.5f, dx,
                         scn.domainBboxMin, scn.domainBboxMax, false,  scn);
        *d_mV = new Grid(d_mVFront, d_mVBack, width, height+1, depth, 0.5f, 0.0f, 0.5f, dx,
                         scn.domainBboxMin, scn.domainBboxMax, true,  scn);
        *d_mW = new Grid(d_mWFront, d_mWBack, width, height, depth+1, 0.5f, 0.5f, 0.0f, dx,
                         scn.domainBboxMin, scn.domainBboxMax, false,  scn);
    }
}

__global__ void free_grids_kernel(Grid **d_mT, Grid **d_mU, Grid **d_mV, Grid **d_mW)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *d_mT;
        delete *d_mU;
        delete *d_mV;
        delete *d_mW;
    }
}

__global__ void clear_back_buffer_kernel(Grid **d_mT, Grid **d_mU, Grid **d_mV, Grid **d_mW) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < (*d_mT)->mWidth && j < (*d_mT)->mHeight && k < (*d_mT)->mDepth) {
        (*d_mT)->atBack(i,j,k) = 0.0f;
    }
    if (i < (*d_mU)->mWidth && j < (*d_mU)->mHeight && k < (*d_mU)->mDepth) {
        (*d_mU)->atBack(i,j,k) = 0.0f;
    }
    if (i < (*d_mV)->mWidth && j < (*d_mV)->mHeight && k < (*d_mV)->mDepth) {
        (*d_mV)->atBack(i,j,k) = 0.0f;
    }
    if (i < (*d_mW)->mWidth && j < (*d_mW)->mHeight && k < (*d_mW)->mDepth) {
        (*d_mW)->atBack(i,j,k) = 0.0f;
    }
}

__global__ void add_source_to_back_buffer_kernel(Grid **d_mT, Grid **d_mU, Grid **d_mV, Grid **d_mW,
                                  float3 *d_mPartPos, float3 *d_mPartVel,
                                  float *d_mPartPscale, float *d_mPartTemp, float dx, int pointCount)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= pointCount) return;

    if (d_mPartPscale[idx] > dx/2.0f) {
        if (abs(d_mPartTemp[idx]) > 1e-5)
            (*d_mT)->sphereToGrid(d_mPartPos[idx].x, d_mPartPos[idx].y, d_mPartPos[idx].z,
                                  d_mPartPscale[idx], d_mPartTemp[idx]);
        if (abs(d_mPartVel[idx].x) > 1e-5)
            (*d_mU)->sphereToGrid(d_mPartPos[idx].x, d_mPartPos[idx].y, d_mPartPos[idx].z,
                                  d_mPartPscale[idx], d_mPartVel[idx].x);
        if (abs(d_mPartVel[idx].y) > 1e-5)
            (*d_mV)->sphereToGrid(d_mPartPos[idx].x, d_mPartPos[idx].y, d_mPartPos[idx].z,
                                  d_mPartPscale[idx], d_mPartVel[idx].y);
        if (abs(d_mPartVel[idx].z) > 1e-5)
            (*d_mW)->sphereToGrid(d_mPartPos[idx].x, d_mPartPos[idx].y, d_mPartPos[idx].z,
                                  d_mPartPscale[idx], d_mPartVel[idx].z);
    }
}

__global__ void set_source_from_back_to_front_kernel(Grid **d_mT, Grid **d_mU, Grid **d_mV, Grid **d_mW) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < (*d_mT)->mWidth && j < (*d_mT)->mHeight && k < (*d_mT)->mDepth && abs((*d_mT)->atBack(i,j,k)) > 1e-5) {
            (*d_mT)->at(i,j,k) = (*d_mT)->atBack(i,j,k);
    }
    if (i < (*d_mU)->mWidth && j < (*d_mU)->mHeight && k < (*d_mU)->mDepth && abs((*d_mU)->atBack(i,j,k)) > 1e-5) {
            (*d_mU)->at(i,j,k) = (*d_mU)->atBack(i,j,k);
    }
    if (i < (*d_mV)->mWidth && j < (*d_mV)->mHeight && k < (*d_mV)->mDepth && abs((*d_mV)->atBack(i,j,k)) > 1e-5) {
            (*d_mV)->at(i,j,k) = (*d_mV)->atBack(i,j,k);
    }
    if (i < (*d_mW)->mWidth && j < (*d_mW)->mHeight && k < (*d_mW)->mDepth && abs((*d_mW)->atBack(i,j,k)) > 1e-5) {
            (*d_mW)->at(i,j,k) = (*d_mW)->atBack(i,j,k);
    }
}

__global__ void temperature_cooldown_kernel(Grid **d_mT, float coolingRate, float dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    (*d_mT)->at(i,j,k) *= (1.0f - dt*coolingRate);
}

__global__ void apply_drag_kernel(Grid **d_mU, Grid **d_mV, Grid **d_mW, float dragRate, float dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < (*d_mU)->mWidth && j < (*d_mU)->mHeight && k < (*d_mU)->mDepth) {
        (*d_mU)->at(i,j,k) *= (1.0f - dt*dragRate);
    }
    if (i < (*d_mV)->mWidth && j < (*d_mV)->mHeight && k < (*d_mV)->mDepth) {
        (*d_mV)->at(i,j,k) *= (1.0f - dt*dragRate);
    }
    if (i < (*d_mW)->mWidth && j < (*d_mW)->mHeight && k < (*d_mW)->mDepth) {
        (*d_mW)->at(i,j,k) *= (1.0f - dt*dragRate);
    }
}

__global__ void add_buoyancy_kernel(Grid **d_mT, Grid **d_mV, float beta, float gravity, float dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < (*d_mV)->mWidth && j < (*d_mV)->mHeight && k < (*d_mV)->mDepth) {
        float3 pos = (*d_mV)->gridToObj(i,j,k);

        float temp = (*d_mT)->sampleO(pos, linear);

        float buoyancyForce = beta*temp*gravity; // gravity is negative

        (*d_mV)->at(i,j,k) += dt*buoyancyForce;
    }
}

__global__ void copy_velocity_to_back_buffer_kernel(Grid **d_mU, Grid **d_mV, Grid **d_mW) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < (*d_mU)->mWidth && j < (*d_mU)->mHeight && k < (*d_mU)->mDepth) {
        (*d_mU)->atBack(i,j,k) = (*d_mU)->at(i,j,k);
    }
    if (i < (*d_mV)->mWidth && j < (*d_mV)->mHeight && k < (*d_mV)->mDepth) {
        (*d_mV)->atBack(i,j,k) = (*d_mV)->at(i,j,k);
    }
    if (i < (*d_mW)->mWidth && j < (*d_mW)->mHeight && k < (*d_mW)->mDepth) {
        (*d_mW)->atBack(i,j,k) = (*d_mW)->at(i,j,k);
    }
}

__device__ float3 computeVorticity(Grid **d_mT, Grid **d_mU, Grid **d_mV, Grid **d_mW, int i, int j, int k, float dx) {
    float x0 = (*d_mT)->iGridToObj(i-1);
    float x1 = (*d_mT)->iGridToObj(i);
    float x2 = (*d_mT)->iGridToObj(i+1);
    float y0 = (*d_mT)->jGridToObj(j-1);
    float y1 = (*d_mT)->jGridToObj(j);
    float y2 = (*d_mT)->jGridToObj(j+1);
    float z0 = (*d_mT)->kGridToObj(k-1);
    float z1 = (*d_mT)->kGridToObj(k);
    float z2 = (*d_mT)->kGridToObj(k+1);

    float w_121 = (*d_mW)->sampleO(make_float3(x1, y2, z1), linear);
    float w_101 = (*d_mW)->sampleO(make_float3(x1, y0, z1), linear);
    float v_112 = (*d_mV)->sampleO(make_float3(x1, y1, z2), linear);
    float v_110 = (*d_mV)->sampleO(make_float3(x1, y1, z0), linear);
    float u_112 = (*d_mU)->sampleO(make_float3(x1, y1, z2), linear);
    float u_110 = (*d_mU)->sampleO(make_float3(x1, y1, z0), linear);
    float w_211 = (*d_mW)->sampleO(make_float3(x2, y1, z1), linear);
    float w_011 = (*d_mW)->sampleO(make_float3(x0, y1, z1), linear);
    float v_211 = (*d_mV)->sampleO(make_float3(x2, y1, z1), linear);
    float v_011 = (*d_mV)->sampleO(make_float3(x0, y1, z1), linear);
    float u_121 = (*d_mU)->sampleO(make_float3(x1, y2, z1), linear);
    float u_101 = (*d_mU)->sampleO(make_float3(x1, y0, z1), linear);

    return make_float3(w_121 - w_101 - (v_112 - v_110),
                       u_112 - u_110 - (w_211 - w_011),
                       v_211 - v_011 - (u_121 - u_101)) / (2.0f*dx);
}

__global__ void vorticity_confinement_kernel(Grid **d_mT, Grid **d_mU, Grid **d_mV, Grid **d_mW,
                                             float conf, float dx, float dt)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i <= 1 || i >= (*d_mT)->mWidth-2 || j <= 1 || j >= (*d_mT)->mHeight-2 || k <= 1 || k >= (*d_mT)->mDepth-2 )
        return;

    float3 omega = computeVorticity(d_mT, d_mU, d_mV, d_mW, i,   j,   k,   dx);

    float3 omega_211 = computeVorticity(d_mT, d_mU, d_mV, d_mW, i+1, j,   k,   dx);
    float3 omega_011 = computeVorticity(d_mT, d_mU, d_mV, d_mW, i-1, j,   k,   dx);
    float3 omega_121 = computeVorticity(d_mT, d_mU, d_mV, d_mW, i,   j+1, k,   dx);
    float3 omega_101 = computeVorticity(d_mT, d_mU, d_mV, d_mW, i,   j-1, k,   dx);
    float3 omega_112 = computeVorticity(d_mT, d_mU, d_mV, d_mW, i,   j,   k+1, dx);
    float3 omega_110 = computeVorticity(d_mT, d_mU, d_mV, d_mW, i,   j,   k-1, dx);

    float3 gradNormOmega = make_float3( length(omega_211) - length(omega_011),
                                        length(omega_121) - length(omega_101),
                                        length(omega_112) - length(omega_110) ) / (2.0f*dx);

    float3 N = gradNormOmega / ( length(gradNormOmega) + 1e-20f*(1.0f/(dx*dt)) );

    // enforce stability by controlling the vorticity magnitude based on local speed
    float3 pos = (*d_mT)->gridToObj(i, j, k);
    float3 vel = make_float3( (*d_mU)->sampleO(pos, linear),
                              (*d_mV)->sampleO(pos, linear),
                              (*d_mW)->sampleO(pos, linear) );
    float speed = length(vel);

    float omegaNorm = length(omega) + 1e-20f*(1.0f/(dx*dt));
    if (omegaNorm > speed) {
        omega = speed * omega / omegaNorm;
    }

    float3 vortConf = conf * dx * cross(gradNormOmega, omega);

    atomicAdd(&((*d_mU)->atBack(i,  j  ,k  )), 0.5f*dt*vortConf.x);
    atomicAdd(&((*d_mU)->atBack(i+1,j  ,k  )), 0.5f*dt*vortConf.x);
    atomicAdd(&((*d_mV)->atBack(i,  j  ,k  )), 0.5f*dt*vortConf.y);
    atomicAdd(&((*d_mV)->atBack(i,  j+1,k  )), 0.5f*dt*vortConf.y);
    atomicAdd(&((*d_mW)->atBack(i,  j  ,k  )), 0.5f*dt*vortConf.z);
    atomicAdd(&((*d_mW)->atBack(i,  j,  k+1)), 0.5f*dt*vortConf.z);
}

__device__ inline float fit(float v, float minIn, float maxIn, float minOut, float maxOut) {
    return fmaxf( fminf( (v-minIn)/(maxIn-minIn), 1.0f),  0.0f)*(maxOut-minOut) + minOut;
}

__device__ float computeTempMask(Grid **grid, Grid **d_mT, int i, int j, int k, float2 maskTempRamp) {
    float3 pos = (*grid)->gridToObj(i,j,k);

    return fit((*d_mT)->sampleO(pos, linear), maskTempRamp.x, maskTempRamp.y, 1.0f, 0.0f);
}

__device__ float computeVelMask(Grid **grid, Grid **d_mU, Grid **d_mV, Grid **d_mW,
                                int i, int j, int k, float2 maskVelRamp) {
    float3 pos = (*grid)->gridToObj(i,j,k);

    float3 vel = make_float3( (*d_mU)->sampleO(pos, linear),
                              (*d_mV)->sampleO(pos, linear),
                              (*d_mW)->sampleO(pos, linear) );

    return fit(length(vel), maskVelRamp.x, maskVelRamp.y, 0.0f, 1.0f);
}

__global__ void add_curl_noise_kernel(Grid **d_mU, Grid **d_mV, Grid **d_mW, float dx, float dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    float dudx = ( (*d_mU)->atBack(i+1, j,   k  ) - (*d_mV)->atBack(i, j, k) ) / dx;
    float dvdy = ( (*d_mV)->atBack(i,   j+1, k  ) - (*d_mV)->atBack(i, j, k) ) / dx;
    float dwdz = ( (*d_mW)->atBack(i,   j,   k+1) - (*d_mV)->atBack(i, j, k) ) / dx;

    float3 curlNoise = make_float3(dvdy-dwdz, dwdz-dudx, dudx-dvdy);

    float avgSpeed = length( make_float3((*d_mU)->atBack(i, j, k) + (*d_mU)->atBack(i+1, j,   k  ),
                                         (*d_mV)->atBack(i, j, k) + (*d_mV)->atBack(i,   j+1, k  ),
                                         (*d_mW)->atBack(i, j, k) + (*d_mW)->atBack(i,   j,   k+1) )/2.0f );
    curlNoise = avgSpeed*curlNoise / (length(curlNoise)+ 1e-20f*(1.0f/(dx*dt)));

    atomicAdd(&((*d_mU)->at(i,  j  ,k  )), 0.5f*dt*curlNoise.x);
    atomicAdd(&((*d_mU)->at(i+1,j  ,k  )), 0.5f*dt*curlNoise.x);
    atomicAdd(&((*d_mV)->at(i,  j  ,k  )), 0.5f*dt*curlNoise.y);
    atomicAdd(&((*d_mV)->at(i,  j+1,k  )), 0.5f*dt*curlNoise.y);
    atomicAdd(&((*d_mW)->at(i,  j  ,k  )), 0.5f*dt*curlNoise.z);
    atomicAdd(&((*d_mW)->at(i,  j,  k+1)), 0.5f*dt*curlNoise.z);
}

__global__ void compute_turbulence_kernel(Grid **d_mT, Grid **d_mU, Grid **d_mV, Grid **d_mW, float amp,
                                          float scale, float2 maskTempRamp, float2 maskVelRamp,
                                          float dt, float time)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    float freq = 1.0f / scale;
    int seed = 123;
    float avgGridSize = ((*d_mT)->mWidth + (*d_mT)->mHeight + (*d_mT)->mDepth) / 3.0f;
    float offset = fmod(time*avgGridSize, 25000.0f); // fmod to avoid floating point roundoff

    if (i < (*d_mU)->mWidth && j < (*d_mU)->mHeight && k < (*d_mU)->mDepth) {
        float tempMaskU = computeTempMask(d_mU, d_mT, i, j, k, maskTempRamp);
        float velMaskU = computeVelMask(d_mU, d_mU, d_mV, d_mW, i, j, k, maskVelRamp);
        float maskU = tempMaskU * velMaskU;

        float3 posU = make_float3((float)i+offset, (float)j, (float)k);
        float noiseU = cudaNoise::perlinNoise(posU, freq, seed);
        (*d_mU)->atBack(i,j,k) = maskU*amp*noiseU;
    }
    if (i < (*d_mV)->mWidth && j < (*d_mV)->mHeight && k < (*d_mV)->mDepth) {
        float tempMaskV = computeTempMask(d_mV, d_mT, i, j, k, maskTempRamp);
        float velMaskV = computeVelMask(d_mV, d_mU, d_mV, d_mW, i, j, k, maskVelRamp);
        float maskV = tempMaskV * velMaskV;

        float3 posV = make_float3((float)i, (float)j+offset, (float)k);
        float noiseV = cudaNoise::perlinNoise(posV, freq, seed);
        (*d_mV)->atBack(i,j,k) = maskV*amp*noiseV;
    }
    if (i < (*d_mW)->mWidth && j < (*d_mW)->mHeight && k < (*d_mW)->mDepth) {
        float tempMaskW = computeTempMask(d_mW, d_mT, i, j, k, maskTempRamp);
        float velMaskW = computeVelMask(d_mW, d_mU, d_mV, d_mW, i, j, k, maskVelRamp);
        float maskW = tempMaskW * velMaskW;

        float3 posW = make_float3((float)i, (float)j, (float)k+offset);
        float noiseW = cudaNoise::perlinNoise(posW, freq, seed);
        (*d_mW)->atBack(i,j,k) = maskW*amp*noiseW;
    }
}

__global__ void add_wind_kernel(Grid **d_mU, Grid **d_mV, Grid **d_mW, int windDir, float windAmp,
                                float windSpeed, float windTurbAmp, float windTurbScale,
                                float dt, float time)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= (*d_mU)->mWidth || j >= (*d_mV)->mHeight || k >= (*d_mW)->mDepth) return;

    int voxelPadding = 2;

    if (windDir <= 1) { // wind along X axis
        if (i >= voxelPadding && i < (*d_mU)->mWidth-voxelPadding) return;

        float freq = 1.0f / windTurbScale;
        float avgDomSize = ((*d_mU)->mDomSize.x + (*d_mU)->mDomSize.y + (*d_mU)->mDomSize.z) / 3.0f;
        float offset = fmod(time*avgDomSize*windSpeed, 25000.0f); // fmod to avoid floating point roundoff

        float3 pos = make_float3((*d_mU)->iGridToObj(i), (*d_mV)->jGridToObj(j), (*d_mW)->kGridToObj(k)+offset);
        float noiseU = windTurbAmp * cudaNoise::perlinNoise(pos, freq, 123);
        float noiseV = windTurbAmp * cudaNoise::perlinNoise(pos, freq, 456);
        float noiseW = windTurbAmp * cudaNoise::perlinNoise(pos, freq, 789);
        float windDirVel = windDir == 0 ? 1.0f : -1.0f;

        // blend in wind vel over 1 second
        if (j < (*d_mU)->mHeight && k < (*d_mU)->mDepth) // U
            (*d_mU)->at(i,j,k) = (*d_mU)->at(i,j,k)*(1.0f-dt) + (windAmp*(windDirVel+noiseU))*dt;
        if (i < (*d_mV)->mWidth && k < (*d_mV)->mDepth)  // V
            (*d_mV)->at(i,j,k) = (*d_mV)->at(i,j,k)*(1.0f-dt) + (windAmp*noiseV)*dt;
        if (i < (*d_mW)->mWidth && j < (*d_mW)->mHeight) // W
            (*d_mW)->at(i,j,k) = (*d_mW)->at(i,j,k)*(1.0f-dt) + (windAmp*noiseW)*dt;
    } else {            // wind along Z axis
        if (k >= voxelPadding && k < (*d_mW)->mDepth-voxelPadding) return;

        float freq = 1.0f / windTurbScale;
        float avgDomSize = ((*d_mU)->mDomSize.x + (*d_mU)->mDomSize.y + (*d_mU)->mDomSize.z) / 3.0f;
        float offset = fmod(time*avgDomSize*windSpeed, 25000.0f); // fmod to avoid floating point roundoff

        float3 pos = make_float3((*d_mU)->iGridToObj(i), (*d_mV)->jGridToObj(j), (*d_mW)->kGridToObj(k)+offset);
        float noiseU = windTurbAmp * cudaNoise::perlinNoise(pos, freq, 123);
        float noiseV = windTurbAmp * cudaNoise::perlinNoise(pos, freq, 456);
        float noiseW = windTurbAmp * cudaNoise::perlinNoise(pos, freq, 789);
        float windDirVel = windDir == 2 ? 1.0f : -1.0f;

        // blend in wind vel over 1 second
        if (j < (*d_mU)->mHeight && k < (*d_mU)->mDepth) // U
            (*d_mU)->at(i,j,k) = (*d_mU)->at(i,j,k)*(1.0f-dt) + (windAmp*noiseU)*dt;
        if (i < (*d_mV)->mWidth && k < (*d_mV)->mDepth)  // V
            (*d_mV)->at(i,j,k) = (*d_mV)->at(i,j,k)*(1.0f-dt) + (windAmp*noiseV)*dt;
        if (i < (*d_mW)->mWidth && j < (*d_mW)->mHeight) // W
            (*d_mW)->at(i,j,k) = (*d_mW)->at(i,j,k)*(1.0f-dt) + (windAmp*(windDirVel+noiseW))*dt;
    }
}

__global__ void advect_forward_euler_kernel(Grid **grid, Grid **d_mU, Grid **d_mV, Grid **d_mW, float dt, bool clamp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < (*grid)->mWidth && j < (*grid)->mHeight && k < (*grid)->mDepth) {
        float3 pos = (*grid)->gridToObj(i,j,k);
        float3 vel = make_float3( (*d_mU)->sampleO(pos, linear),
                                  (*d_mV)->sampleO(pos, linear),
                                  (*d_mW)->sampleO(pos, linear) );

        pos -= dt*vel;

        (*grid)->atBack(i,j,k) = (*grid)->sampleO(pos, cubic, clamp);
    }
}

__global__ void advect_RK2_kernel(Grid **grid, Grid **d_mU, Grid **d_mV, Grid **d_mW, float dt, bool clamp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < (*grid)->mWidth && j < (*grid)->mHeight && k < (*grid)->mDepth) {
        float3 pos = (*grid)->gridToObj(i,j,k);
        float3 vel = make_float3( (*d_mU)->sampleO(pos, linear),
                                  (*d_mV)->sampleO(pos, linear),
                                  (*d_mW)->sampleO(pos, linear) );

        float3 posMid = pos - 0.5f*dt*vel;
        float3 velMid = make_float3( (*d_mU)->sampleO(posMid, linear),
                                     (*d_mV)->sampleO(posMid, linear),
                                     (*d_mW)->sampleO(posMid, linear) );

        pos -= dt*velMid;

        (*grid)->atBack(i,j,k) = (*grid)->sampleO(pos, cubic, clamp);
    }
}

__global__ void advect_RK3_kernel(Grid **grid, Grid **d_mU, Grid **d_mV, Grid **d_mW, float dt, bool clamp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < (*grid)->mWidth && j < (*grid)->mHeight && k < (*grid)->mDepth) {
        float3 pos = (*grid)->gridToObj(i,j,k);
        float3 k1 = make_float3( (*d_mU)->sampleO(pos, linear),
                                 (*d_mV)->sampleO(pos, linear),
                                 (*d_mW)->sampleO(pos, linear) );

        float3 pos1 = pos - 0.5f*dt*k1;
        float3 k2 = make_float3( (*d_mU)->sampleO(pos1, linear),
                                 (*d_mV)->sampleO(pos1, linear),
                                 (*d_mW)->sampleO(pos1, linear) );

        float3 pos2 = pos - 0.75f*dt*k2;
        float3 k3 = make_float3( (*d_mU)->sampleO(pos2, linear),
                                 (*d_mV)->sampleO(pos2, linear),
                                 (*d_mW)->sampleO(pos2, linear) );

        pos -= dt * ( (2.0f/9.0f)*k1 + (3.0f/9.0f)*k2 + (4.0f/9.0f)*k3 );

        (*grid)->atBack(i,j,k) = (*grid)->sampleO(pos, cubic, clamp);
    }
}

__global__ void swap_grids_kernel(Grid **d_mT, Grid **d_mU, Grid **d_mV, Grid **d_mW)  {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*d_mT)->swap();
        (*d_mU)->swap();
        (*d_mV)->swap();
        (*d_mW)->swap();
    }
}

__global__ void swap_vel_grids_kernel(Grid **d_mU, Grid **d_mV, Grid **d_mW)  {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*d_mU)->swap();
        (*d_mV)->swap();
        (*d_mW)->swap();
    }
}

__global__ void compute_divergence_kernel(float *d_mRhs, Grid **d_mU, Grid **d_mV, Grid **d_mW,
                                          int width, int height, int depth, float density, float dx)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

//    assert(i < width);
//    assert(j < height);
//    assert(k < depth);

    int idx = i + j*width + k*width*height;
    float scale = 1.0f / dx;

    d_mRhs[idx] = -scale*( (*d_mU)->at(i+1,j,k) - (*d_mU)->at(i,j,k)
                         + (*d_mV)->at(i,j+1,k) - (*d_mV)->at(i,j,k)
                         + (*d_mW)->at(i,j,k+1) - (*d_mW)->at(i,j,k));

    // handle boundary condition (closed or opened bounds)
    if (i == 0) {               // xMin
        if ((*d_mU)->mClosedBounds[0])
            d_mRhs[idx] -= scale*(*d_mU)->at(i,j,k);
    } else if (i == width-1) {  // xMax
        if ((*d_mU)->mClosedBounds[1])
            d_mRhs[idx] += scale*(*d_mU)->at(i+1,j,k);
    }
    if (j == 0) {               // yMax (don't forget the y is flipped)
        if ((*d_mV)->mClosedBounds[3])
            d_mRhs[idx] -= scale*(*d_mV)->at(i,j,k);
    } else if (j == height-1) { // yMin (don't forget the y is flipped)
        if ((*d_mV)->mClosedBounds[2])
            d_mRhs[idx] += scale*(*d_mV)->at(i,j+1,k);
    }
    if (k == 0) {               // zMin
        if ((*d_mW)->mClosedBounds[4])
            d_mRhs[idx] -= scale*(*d_mW)->at(i,j,k);
    } else if (k == depth-1) {  // zMax
        if ((*d_mW)->mClosedBounds[5])
            d_mRhs[idx] += scale*(*d_mW)->at(i,j,k+1);
    }
}

__global__ void gs_solve_kernel(float* rhs, float* p, float density, float dx, float dt,
                                int width, int height, int depth, int maxIter)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

//    assert(i < width);
//    assert(j < height);

    for (int l=0; l<maxIter; l++) {
        for (int k=0; k<depth; k++) {

            int idx = i + j*width + k*width*height;
            float scale = (density*dx*dx)/dt; // Bridson p.75

            if( (i + j)%2 == 0 ) {
                float denom = 0.0f;
                float num = scale*rhs[idx];

                if (i > 0) {
                    num += p[idx - 1];
                    denom += 1.0f;
                }
                if (i < width-1) {
                    num += p[idx + 1];
                    denom += 1.0f;
                }
                if (j > 0) {
                    num += p[idx - width];
                    denom += 1.0f;
                }
                if (j < height-1) {
                    num += p[idx + width];
                    denom += 1.0f;
                }
                if (k > 0) {
                    num += p[idx - width*height];
                    denom += 1.0f;
                }
                if (k < depth-1) {
                    num += p[idx + width*height];
                    denom += 1.0f;
                }

                p[idx] = num / denom;
            }

            __syncthreads();

            if( (i + j)%2 != 0 ) {
                float denom = 0.0f;
                float num = scale*rhs[idx];

                if (i > 0) {
                    num += p[idx - 1];
                    denom += 1.0f;
                }
                if (i < width-1) {
                    num += p[idx + 1];
                    denom += 1.0f;
                }
                if (j > 0) {
                    num += p[idx - width];
                    denom += 1.0f;
                }
                if (j < height-1) {
                    num += p[idx + width];
                    denom += 1.0f;
                }
                if (k > 0) {
                    num += p[idx - width*height];
                    denom += 1.0f;
                }
                if (k < depth-1) {
                    num += p[idx + width*height];
                    denom += 1.0f;
                }

                p[idx] = num / denom;
            }
        }
    }
}

__global__ void pressure_gradient_update_kernel(float *d_mP, Grid **d_mU, Grid **d_mV, Grid **d_mW,
                                                int width, int height, int depth, float density,
                                                float dx, float dt)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= width+1 || j >= height+1 || k >= depth+1) return;

    int idx = i + j*width + k*width*height;
    float scale = dt / (density*dx);

    if (i < width && j < height && k < depth) {
        if (i > 0 ) {
            (*d_mU)->at(i, j, k) -= scale*(d_mP[idx] - d_mP[idx - 1]);
        }
        if (j > 0 ) {
            (*d_mV)->at(i, j, k) -= scale*(d_mP[idx] - d_mP[idx - width]);
        }
        if (k > 0 ) {
            (*d_mW)->at(i, j, k) -= scale*(d_mP[idx] - d_mP[idx - width*height]);
        }
    }

    // set the boundary velocities (closed or opened bounds)
    // The lines marked with a "?" indicates that it should be deleted according
    // to p.71 of Bridson's book. Our tests shows it is needed, though...
    // Free surface implementation might be containing a bug.
    if ( (i == 0 || i == width) && j < height && k < depth) { // X
        if (i == 0) { // xMin
            if ((*d_mU)->mClosedBounds[0])
                (*d_mU)->at(i, j, k) = 0.0f;
            else
                (*d_mU)->at(i, j, k) = (*d_mU)->at(i+1, j, k); // ?
        } else {      // xMax
             if ((*d_mU)->mClosedBounds[1])
                 (*d_mU)->at(i, j, k) = 0.0f;
             else
                 (*d_mU)->at(i, j, k) = (*d_mU)->at(i-1, j, k); // ?
        }
    }
    if ( (j == 0 || j == height) && i < width && k < depth) { // Y
        if (j == 0) { // yMax (don't forget the y is flipped)
            if ((*d_mV)->mClosedBounds[3])
                (*d_mV)->at(i, j, k) = 0.0f;
            else
                (*d_mV)->at(i, j, k) = (*d_mV)->at(i, j+1, k); // ?
        } else {      // yMin (don't forget the y is flipped)
             if ((*d_mV)->mClosedBounds[2])
                 (*d_mV)->at(i, j, k) = 0.0f;
             else
                 (*d_mV)->at(i, j, k) = (*d_mV)->at(i, j-1, k); // ?
        }
    }
    if ( (k == 0 || k == depth) && i < width && j < height) { // Z
        if (k == 0) { // zMin
            if ((*d_mW)->mClosedBounds[4])
                (*d_mW)->at(i, j, k) = 0.0f;
            else
                (*d_mW)->at(i, j, k) = (*d_mW)->at(i, j, k+1); // ?
        } else {      // zMax
             if ((*d_mW)->mClosedBounds[5])
                 (*d_mW)->at(i, j, k) = 0.0f;
             else
                 (*d_mW)->at(i, j, k) = (*d_mW)->at(i, j, k-1); // ?
        }
    }
}

FluidSolver::FluidSolver(Timer *tmr, SceneSettings *scn)
    : mWidth(scn->gridRes.x)
    , mHeight(scn->gridRes.y)
    , mDepth(scn->gridRes.z)
    , mDensity(scn->density)
    , mMaxIter(scn->maxIterSolve)
    , mDt(scn->dt)
    , mDx(scn->dx)
    , mTmr(tmr)
    , mBuoyancy(scn->buoyancy)
    , mCoolingRate(scn->coolingRate)
    , mGravity(scn->gravity)
    , mVorticeConf(scn->vorticityConf)
    , mDrag(scn->drag)
    , mTurbulenceAmp(scn->turbulence_amp)
    , mTurbulenceScale(scn->turbulence_scale)
    , mTurbMaskTempRamp(scn->turbMaskTempRamp)
    , mTurbMaskVelRamp(scn->turbMaskVelRamp)
    , mScn(scn)
    , mSingleFrameSourceInit(false)
    , mParticleCount(0)
    , mByteSize ( scn->gridRes.x    *  scn->gridRes.y    *  scn->gridRes.z    * sizeof(float))
    , mByteSizeU((scn->gridRes.x+1) *  scn->gridRes.y    *  scn->gridRes.z    * sizeof(float))
    , mByteSizeV( scn->gridRes.x    * (scn->gridRes.y+1) *  scn->gridRes.z    * sizeof(float))
    , mByteSizeW( scn->gridRes.x    *  scn->gridRes.y    * (scn->gridRes.z+1) * sizeof(float))
{
    checkCudaErrors(cudaMalloc((void**)&d_mTFront, mByteSize));
    checkCudaErrors(cudaMalloc((void**)&d_mTBack, mByteSize));
    checkCudaErrors(cudaMalloc((void**)&d_mUFront, mByteSizeU));
    checkCudaErrors(cudaMalloc((void**)&d_mUBack, mByteSizeU));
    checkCudaErrors(cudaMalloc((void**)&d_mVFront, mByteSizeV));
    checkCudaErrors(cudaMalloc((void**)&d_mVBack, mByteSizeV));
    checkCudaErrors(cudaMalloc((void**)&d_mWFront, mByteSizeW));
    checkCudaErrors(cudaMalloc((void**)&d_mWBack, mByteSizeW));
    checkCudaErrors(cudaMemset(d_mTFront, 0, mByteSize));
    checkCudaErrors(cudaMemset(d_mTBack, 0, mByteSize));
    checkCudaErrors(cudaMemset(d_mUFront, 0, mByteSizeU));
    checkCudaErrors(cudaMemset(d_mUBack, 0, mByteSizeU));
    checkCudaErrors(cudaMemset(d_mVFront, 0, mByteSizeV));
    checkCudaErrors(cudaMemset(d_mVBack, 0, mByteSizeV));
    checkCudaErrors(cudaMemset(d_mWFront, 0, mByteSizeW));
    checkCudaErrors(cudaMemset(d_mWBack, 0, mByteSizeW));

    checkCudaErrors(cudaMalloc((void**)&d_mRhs, mByteSize));
    checkCudaErrors(cudaMalloc((void**)&d_mP, mByteSize));
    checkCudaErrors(cudaMemset(d_mP, 0, mByteSize));

    // create grids object on device
    checkCudaErrors(cudaMalloc((void **)&d_mT, sizeof(Grid *)));
    checkCudaErrors(cudaMalloc((void **)&d_mU, sizeof(Grid *)));
    checkCudaErrors(cudaMalloc((void **)&d_mV, sizeof(Grid *)));
    checkCudaErrors(cudaMalloc((void **)&d_mW, sizeof(Grid *)));
    create_grids_kernel<<<1,1>>>(d_mT, d_mU, d_mV, d_mW, d_mTFront, d_mTBack, d_mUFront, d_mUBack,
                                 d_mVFront, d_mVBack, d_mWFront, d_mWBack, mWidth, mHeight, mDepth, mDx, *scn);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // allocate particle arrays on host and device
    int byteSizePartFloat3 = scn->sourceMaxParticleCount*sizeof(float3);
    int byteSizePartFloat = scn->sourceMaxParticleCount*sizeof(float);

    h_mPartPos = (float3*)malloc(byteSizePartFloat3);
    h_mPartVel = (float3*)malloc(byteSizePartFloat3);
    h_mPartPscale = (float*)malloc(byteSizePartFloat);
    h_mPartTemp = (float*)malloc(byteSizePartFloat);

    checkCudaErrors(cudaMalloc((void**)&d_mPartPos, byteSizePartFloat3));
    checkCudaErrors(cudaMalloc((void**)&d_mPartVel, byteSizePartFloat3));
    checkCudaErrors(cudaMalloc((void**)&d_mPartPscale, byteSizePartFloat));
    checkCudaErrors(cudaMalloc((void**)&d_mPartTemp, byteSizePartFloat));
}

FluidSolver::~FluidSolver() {
    // grids arrays
    free_grids_kernel<<<1,1>>>(d_mT, d_mU, d_mV, d_mW);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_mTFront));
    checkCudaErrors(cudaFree(d_mTBack));
    checkCudaErrors(cudaFree(d_mUFront));
    checkCudaErrors(cudaFree(d_mUBack));
    checkCudaErrors(cudaFree(d_mVFront));
    checkCudaErrors(cudaFree(d_mVBack));
    checkCudaErrors(cudaFree(d_mWFront));
    checkCudaErrors(cudaFree(d_mWBack));

    checkCudaErrors(cudaFree(d_mRhs));
    checkCudaErrors(cudaFree(d_mP));

    // particle arrays
    checkCudaErrors(cudaFree(d_mPartPos));
    checkCudaErrors(cudaFree(d_mPartVel));
    checkCudaErrors(cudaFree(d_mPartPscale));
    checkCudaErrors(cudaFree(d_mPartTemp));
    free(h_mPartPos);
    free(h_mPartVel);
    free(h_mPartPscale);
    free(h_mPartTemp);
}

void FluidSolver::addSource() {

    mTmr->source_in = clock();

    // if sourcing is not animated it will happen on every frame
    if (!mScn->sourceAnimated || (mScn->sourceAnimated &&
                                  mTmr->iter >= mScn->sourceRange.x &&
                                  mTmr->iter <= mScn->sourceRange.y))
    {
        if (!mSingleFrameSourceInit || mScn->sourceAnimated) {
            // parse custom particles file and fill host particle array
            Parser::sourceParticleParse(mScn, h_mPartPos, h_mPartVel, h_mPartPscale, h_mPartTemp, mParticleCount, mTmr->iter);

            if (mParticleCount > 0) {
                // copy particle arrays to device
                int byteSizePartFloat3 = mParticleCount*sizeof(float3);
                int byteSizePartFloat = mParticleCount*sizeof(float);
                checkCudaErrors(cudaMemcpy(d_mPartPos, h_mPartPos, byteSizePartFloat3, cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpy(d_mPartVel, h_mPartVel, byteSizePartFloat3, cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpy(d_mPartPscale, h_mPartPscale, byteSizePartFloat, cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpy(d_mPartTemp, h_mPartTemp, byteSizePartFloat, cudaMemcpyHostToDevice));
            }

            mSingleFrameSourceInit = true;
        }

        if (mParticleCount > 0) {
            // clear back buffers
            dim3 block(8, 8, 8);
            dim3 grid((mWidth+1)/block.x+1, (mHeight+1)/block.y+1, (mDepth+1)/block.z+1);
            clear_back_buffer_kernel <<< grid, block >>> (d_mT, d_mU, d_mV, d_mW);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            // scatter particle-to-grid to back buffer
            dim3 blockPart(32, 1, 1);
            dim3 gridPart(mParticleCount/blockPart.x+1, 1, 1);
            add_source_to_back_buffer_kernel <<< gridPart, blockPart >>> (d_mT, d_mU, d_mV, d_mW, d_mPartPos, d_mPartVel,
                                                                          d_mPartPscale, d_mPartTemp, mDx, mParticleCount);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            // copy value from back buffer to front buffer
            set_source_from_back_to_front_kernel <<< grid, block >>> (d_mT, d_mU, d_mV, d_mW);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }
    }

    mTmr->source_out = clock();

}

void FluidSolver::project() {

    dim3 block(8, 8, 8);
    dim3 grid(mWidth/block.x, mHeight/block.y, mDepth/block.z);

    mTmr->computeDivergence_in = clock();

    compute_divergence_kernel <<< grid, block >>> (d_mRhs, d_mU, d_mV, d_mW, mWidth, mHeight,
                                                   mDepth, mDensity, mDx);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    mTmr->computeDivergence_out = clock();

    mTmr->gsSolve_in = clock();

    dim3 blockGS(16, 16, 1);
    dim3 gridGS(mWidth/blockGS.x, mHeight/blockGS.y, 1);
    gs_solve_kernel <<< gridGS, blockGS >>> (d_mRhs, d_mP, mDensity, mDx, mDt, mWidth, mHeight, mDepth, mMaxIter);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    mTmr->gsSolve_out = clock();

    mTmr->pressureGradientUpdate_in = clock();

    dim3 gridP((mWidth+1)/block.x+1, (mHeight+1)/block.y+1, (mDepth+1)/block.z+1); // padding for the staggered velocity bounds
    pressure_gradient_update_kernel <<< gridP, block >>> (d_mP, d_mU, d_mV, d_mW, mWidth, mHeight,
                                                          mDepth, mDensity, mDx, mDt);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    mTmr->pressureGradientUpdate_out = clock();
}

void FluidSolver::step () {
    dim3 block(8, 8, 8);
    dim3 gridT(mWidth     /block.x,     mHeight   /block.y,      mDepth   /block.z    );
    dim3 gridU((mWidth+1) /block.x + 1, mHeight   /block.y,      mDepth   /block.z    );
    dim3 gridV(mWidth     /block.x,    (mHeight+1)/block.y + 1,  mDepth   /block.z    );
    dim3 gridW(mWidth     /block.x,     mHeight   /block.y,     (mDepth+1)/block.z + 1);
    dim3 gridUVW(mWidth+1 /block.x + 1,(mHeight+1)/block.y + 1, (mDepth+1)/block.z + 1);

    mTmr->cooldown_in = clock();

    if (abs(mCoolingRate) > 0.0f) {
        temperature_cooldown_kernel <<< gridT, block >>> (d_mT, mCoolingRate, mDt);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    mTmr->cooldown_out = clock();

    mTmr->drag_in = clock();

    if (mDrag > 0.0f) {
        apply_drag_kernel <<< gridUVW, block >>> (d_mU, d_mV, d_mW, mDrag, mDt);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    mTmr->drag_out = clock();

    mTmr->vorticity_in = clock();

    if (abs(mVorticeConf) > 0.0f) {
        copy_velocity_to_back_buffer_kernel <<< gridUVW, block >>> (d_mU, d_mV, d_mW);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        vorticity_confinement_kernel <<< gridT, block >>> (d_mT, d_mU, d_mV, d_mW, mVorticeConf, mDx, mDt);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        swap_vel_grids_kernel <<< 1, 1 >>> (d_mU, d_mV, d_mW);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    mTmr->vorticity_out = clock();

    mTmr->buoyancy_in = clock();

    if (abs(mBuoyancy) > 0.0f) {
        add_buoyancy_kernel <<< gridV, block >>> (d_mT, d_mV, mBuoyancy, mGravity, mDt);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    mTmr->buoyancy_out = clock();

    mTmr->wind_in = clock();

    if (mScn->windAmp > 0.0f) {
        add_wind_kernel <<< gridUVW, block >>> (d_mU, d_mV, d_mW, mScn->windDir, mScn->windAmp, mScn->windSpeed,
                                                mScn->windTurbAmp, mScn->windTurbScale, mDt, mTmr->time);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    mTmr->wind_out = clock();

    mTmr->turbulence_in = clock();

    if (mTurbulenceAmp > 0.0f) {
        compute_turbulence_kernel <<< gridUVW, block >>> (d_mT, d_mU, d_mV, d_mW, mTurbulenceAmp, mTurbulenceScale,
                                                          mTurbMaskTempRamp, mTurbMaskVelRamp, mDt, mTmr->time);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        add_curl_noise_kernel <<< gridT, block >>> (d_mU, d_mV, d_mW, mDx, mDt);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    mTmr->turbulence_out = clock();

    project();

    mTmr->advect_in = clock();

    advect_RK3_kernel <<< gridT, block >>> (d_mT , d_mU, d_mV, d_mW, mDt, true);
    advect_RK3_kernel <<< gridU, block >>> (d_mU , d_mU, d_mV, d_mW, mDt, false);
    advect_RK3_kernel <<< gridV, block >>> (d_mV , d_mU, d_mV, d_mW, mDt, false);
    advect_RK3_kernel <<< gridW, block >>> (d_mW , d_mU, d_mV, d_mW, mDt, false);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    mTmr->advect_out = clock();

    swap_grids_kernel <<< 1, 1 >>> (d_mT, d_mU, d_mV, d_mW);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

}
