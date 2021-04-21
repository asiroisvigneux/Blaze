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


#include "../thirdparty/lodepng/lodepng.h"
#define TINYEXR_IMPLEMENTATION
#include "../thirdparty/tinyexr/tinyexr.h"
#include "utils/helper_math.h"

#include "RenderEngine.h"


__constant__ float4 c_filterData[3*3*3];

const float RenderEngine::mFilterBlur[RenderEngine::mFilterSize] =
{
    0,1,0,
    1,2,1,
    0,1,0,

    1,2,1,
    2,4,2,
    1,2,1,

    0,1,0,
    1,2,1,
    0,1,0,
};

__global__ void gaussian_blur_texture_kernel(cudaTextureObject_t volumeTexIn,
                                             cudaSurfaceObject_t volumeTexOut,
                                             int filterSize, cudaExtent volumeSize)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (!(i < volumeSize.width && j < volumeSize.height && k < volumeSize.depth)) return;

    float filtered = 0;
    float4 basecoord = make_float4(i+0.5f, j+0.5f, k+0.5f, 0);

    for (int i=0; i<filterSize; i++)
    {
        float4 coord = basecoord + c_filterData[i];
        filtered  += tex3D<float>(volumeTexIn, coord.x, coord.y, coord.z) * c_filterData[i].w;
    }

    // surface writes need byte offsets for x!
    surf3Dwrite(filtered, volumeTexOut, i*sizeof(float), j, k);
}

__global__ void write_scatter_to_3d_tex_kernel(Grid **tempGrid, cudaTextureObject_t scatterTex,
                                               float scatterTempMin, int resRatio, cudaExtent volumeSize)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (!(i < volumeSize.width && j < volumeSize.height && k < volumeSize.depth)) return;

    // downscale 64 temp voxel to 1 scatter voxel
    float output = 0.0f;
    for (int io=0; io<resRatio; io++) {
        for (int jo=0; jo<resRatio; jo++) {
            for (int ko=0; ko<resRatio; ko++) {
                float t = (*tempGrid)->at(i*resRatio+io, j*resRatio+jo, k*resRatio+ko);
                output += t>scatterTempMin?t:0.0f;
            }
        }
    }
    output /= (float)(resRatio*resRatio*resRatio);

    // surface writes need byte offsets for x!
    surf3Dwrite(output, scatterTex, i*sizeof(float), j, k);
}

__global__ void write_temp_to_3d_tex_kernel(Grid **tempGrid, cudaTextureObject_t tempTex)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

//    assert(i < (*tempGrid)->mWidth && j < (*tempGrid)->mHeight && k < (*tempGrid)->mDepth);

    float output = (*tempGrid)->at(i,j,k);

    // surface writes need byte offsets for x!
    surf3Dwrite(output, tempTex, i*sizeof(float), j, k);
}

// https://www.scratchapixel.com/
__device__ bool bbox_intersect_kernel(Ray &r, float3 bounds[2]) {
    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    tmin = (bounds[r.mSign.x].x - r.mOrig.x) * r.mInvdir.x;
    tmax = (bounds[1-r.mSign.x].x - r.mOrig.x) * r.mInvdir.x;
    tymin = (bounds[r.mSign.y].y - r.mOrig.y) * r.mInvdir.y;
    tymax = (bounds[1-r.mSign.y].y - r.mOrig.y) * r.mInvdir.y;

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    tzmin = (bounds[r.mSign.z].z - r.mOrig.z) * r.mInvdir.z;
    tzmax = (bounds[1-r.mSign.z].z - r.mOrig.z) * r.mInvdir.z;

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;

    r.mT0 = fmaxf(tmin, 0.0f);
    r.mT1 = tmax;

    if (r.mT1 < 0.0f)
        return false;

    return true;
}

__global__ void render_kernel(float3 *frameBuffer, Grid **tempGrid,
                              cudaTextureObject_t tempTex,
                              cudaTextureObject_t scatterTex,
                              Shader **shader, Light **lights, int lightCount,
                              Camera **cam, float dx, int scatterResRatio,
                              float pStep, float sStep, float cutoff)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    // initialize the pixels to black
    frameBuffer[i + (*cam)->mRes.x*j] = make_float3(0.0f);

    Ray pRay = (*cam)->getRay(i,j);
    float3 bounds[] = {(*tempGrid)->mDomMin, (*tempGrid)->mDomMax};

    bool hit = bbox_intersect_kernel(pRay, bounds);
    if (!hit) return;

    float3 pLumi = make_float3(0.0f);
    float3 pTrans = make_float3(1.0f);
    float3 one = make_float3(1.0f);

    // primary ray loop
    for (float pT = pRay.mT0+dx*0.5; pT <= pRay.mT1; pT += pStep) {

        float3 pPos = pRay.at(pT);
        float3 pPosGrid = (*tempGrid)->worldToGridNoOffset( pPos );

        const float pTemp = tex3D<float>(tempTex, pPosGrid.x, pPosGrid.y, pPosGrid.z);
        const float pScatter = tex3D<float>(scatterTex, pPosGrid.x/(float)scatterResRatio,
                                                        pPosGrid.y/(float)scatterResRatio,
                                                        pPosGrid.z/(float)scatterResRatio);
        const float pDensity = pTemp * (*shader)->mDensityScale;

        if (pDensity < cutoff) continue;

        float3 emission = make_float3(0.0f);
        if ((*shader)->mEmissionScale > 0.0f) {
            emission = (*shader)->getEmissionColor(pTemp) * (*shader)->mEmissionScale
                     + (*shader)->mMultiScatterColor * pScatter * (*shader)->mMultiScatterScale
                     * smoothstep((*shader)->mMultiScatterDensityMask.x, (*shader)->mMultiScatterDensityMask.y, pTemp);
        }

        // compute the delta transmitance using Lambert-Beers law (P.176 PVR)
        const float3 dT = expf((*shader)->mExtinction * pDensity * pStep);

        // light loop
        for (int i=0; i<lightCount; i++) {
            float3 lightDir, lightIntensity;
            lights[i]->illuminate(pPos, lightDir, lightIntensity);

            Ray sRay(pPos, -1.0f*lightDir);
            hit = bbox_intersect_kernel(sRay, bounds);

            float3 sTrans = make_float3(1.0f);

            // secondary ray loop
            for (float sT = sRay.mT0+dx*0.5; sT <= sRay.mT1; sT += sStep) {
                float3 sPosGrid = (*tempGrid)->worldToGridNoOffset( sRay.at(sT) );
                const float sDensity = tex3D<float>(tempTex, sPosGrid.x, sPosGrid.y, sPosGrid.z) * (*shader)->mDensityScale * (*shader)->mShadowDensityScale;
                if (sDensity < cutoff) continue;
                sTrans *= expf((*shader)->mExtinction * sDensity * sStep/(1.0f+sT*(*shader)->mSGain));
                if (squared_length(sTrans) < cutoff) break;
            }

            pLumi += (*shader)->mAlbedo * sTrans * pTrans * lightIntensity * (one-dT);
        }
        pLumi += pTrans * emission * (one-dT);
        pTrans *= dT;

        if (squared_length(pTrans) < cutoff) break;
    }

    frameBuffer[i + (*cam)->mRes.x*j] = pLumi;

}

__global__ void create_scene_kernel(Camera **d_mCamera, Shader **d_mShader, Light **d_mLights,
                                    SceneSettings scn, float dx)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_mCamera = new Camera(scn.camTrans, scn.camU, scn.camV, scn.camW, scn.camFocal,
                                scn.camAperture, scn.renderRes.x, scn.renderRes.y);
        *d_mShader = new Shader(scn);
        for (int i=0; i<scn.lightCount; i++) {
            d_mLights[i] = new Light(scn, i);
        }
    }
}

__global__ void delete_scene_kernel(Light **d_mLights, int lightCount,
                                    Shader **d_mShader, Camera **d_mCamera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *d_mShader;
        delete *d_mCamera;
        for (int i=0; i<lightCount; i++) {
            delete d_mLights[i];
        }
    }
}

RenderEngine::RenderEngine(Timer *tmr, SceneSettings scn)
    : mTmr(tmr)
    , mWidth(scn.gridRes.x)
    , mHeight(scn.gridRes.y)
    , mDepth(scn.gridRes.z)
    , mRdrWidth(scn.renderRes.x)
    , mRdrHeight(scn.renderRes.y)
    , mScatterScale(scn.multiScatterScale)
    , mScatterTempMin(scn.shaderTempRange.x)
    , mScatterBlurIter(scn.multiScatterBlurIter)
    , mDx(scn.dx)
    , mLightCount(scn.lightCount)
    , mPStep(scn.rndrPStep)
    , mSStep(scn.rndrSStep)
    , mCutoff(scn.rndrCutoff)
    , mRenderFile(scn.renderFile)
    , mSceneDir(scn.sceneDir)
{
    int byteSizeFloat3 = mRdrWidth*mRdrHeight*sizeof(float3);

    // allocate frameBuffer shared between CPU and GPU
    checkCudaErrors(cudaMallocManaged((void **)&mFrameBuffer, byteSizeFloat3));

    checkCudaErrors(cudaMalloc((void **)&d_mShader, sizeof(Shader *)));
    checkCudaErrors(cudaMalloc((void **)&d_mCamera, sizeof(Camera *)));
    checkCudaErrors(cudaMalloc((void **)&d_mLights, scn.lightCount*sizeof(Light *)));
    create_scene_kernel <<< 1, 1 >>> (d_mCamera, d_mShader, d_mLights, scn, mDx);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    mTempVol = new Volume(mWidth, mHeight, mDepth);
    mScatterFrontVol = new Volume(mWidth/mScatterResRatio, mHeight/mScatterResRatio, mDepth/mScatterResRatio);
    mScatterBackVol = new Volume(mWidth/mScatterResRatio, mHeight/mScatterResRatio, mDepth/mScatterResRatio);

    // setup gaussian blur filter weights
    float4 weights[mFilterSize];

    float sum = 0;
    for (int i=0; i<mFilterSize; i++) sum += mFilterBlur[i];

    int idx = 0;
    for (int k=-mFilterWidth/2; k<mFilterWidth/2+1; k++) {
        for (int j=-mFilterWidth/2; j<mFilterWidth/2+1; j++) {
            for (int i=-mFilterWidth/2; i<mFilterWidth/2+1; i++, idx++) {
                weights[idx] = make_float4(i, j, k, mFilterBlur[idx]/sum);
            }
        }
    }
    checkCudaErrors(cudaMemcpyToSymbol(c_filterData, weights, sizeof(float4)*mFilterSize));
}

RenderEngine::~RenderEngine() {
    delete mTempVol;
    delete mScatterFrontVol;
    delete mScatterBackVol;

    checkCudaErrors(cudaFree(mFrameBuffer));
    delete_scene_kernel <<< 1, 1 >>> (d_mLights, mLightCount, d_mShader, d_mCamera);
    checkCudaErrors(cudaGetLastError());
}

void RenderEngine::render(Grid **tempGrid) {

    // write temp to 3d texture
    dim3 blockSize(8,8,8);
    dim3 gridSize(mWidth/blockSize.x, mHeight/blockSize.y, mDepth/blockSize.z);
    write_temp_to_3d_tex_kernel <<< gridSize, blockSize >>> (tempGrid, mTempVol->volumeSurf);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    mTmr->scatter_in = clock();

    if (mScatterScale > 0.0f) {
        dim3 gridSizeScatter((mWidth/mScatterResRatio) /blockSize.x+1,
                             (mHeight/mScatterResRatio)/blockSize.y+1,
                             (mDepth/mScatterResRatio) /blockSize.z+1 );
        write_scatter_to_3d_tex_kernel <<< gridSizeScatter, blockSize >>> (tempGrid,
                                                                           mScatterFrontVol->volumeSurf,
                                                                           mScatterTempMin,
                                                                           mScatterResRatio,
                                                                           mScatterFrontVol->size);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        Volume *swap = 0;
        for (int i=0; i<mScatterBlurIter; i++) {
            gaussian_blur_texture_kernel <<< gridSizeScatter, blockSize >>> (mScatterFrontVol->volumeTex,
                                                                             mScatterBackVol->volumeSurf,
                                                                             mFilterSize, mScatterFrontVol->size);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            // swap textures for iterative blur
            swap = mScatterFrontVol;
            mScatterFrontVol = mScatterBackVol;
            mScatterBackVol = swap;
        }
    }

    mTmr->scatter_out = clock();

    mTmr->render_in = clock();

    // rendered image should be multiple of 8
    dim3 block(8, 8, 1);
    dim3 grid(mRdrWidth/block.x, mRdrHeight/block.y, 1);
    render_kernel <<< grid, block >>> (mFrameBuffer, tempGrid, mTempVol->volumeTex, mScatterFrontVol->volumeTex,
                                       d_mShader, d_mLights, mLightCount, d_mCamera, mDx, mScatterResRatio,
                                       mPStep, mSStep, mCutoff);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    mTmr->render_out = clock();

}

void RenderEngine::writeToDisk() {

    mTmr->writeToDisk_in = clock();

    char filename[256];
    std::string stringRenderFile(mRenderFile);
    std::string subStrName = stringRenderFile.substr(0, stringRenderFile.length()-9);
    std::string subStrExt = stringRenderFile.substr(stringRenderFile.length()-3, stringRenderFile.length());
    sprintf(filename, "%s/%s_%04d.%s", mSceneDir.c_str(), subStrName.c_str(), mTmr->iter, subStrExt.c_str());

    std::string renderDir(Parser::getDirPath(filename));
    mkdir(renderDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    if (subStrExt == "ppm") {

        std::ofstream ofs(filename, std::ios_base::out | std::ios_base::binary);
        ofs << "P3\n" << mRdrWidth << " " << mRdrHeight << "\n255\n";
        for (int i=0; i<mRdrWidth*mRdrHeight; i++) {
            int ir = int(255.99*mFrameBuffer[i].x);
            int ig = int(255.99*mFrameBuffer[i].y);
            int ib = int(255.99*mFrameBuffer[i].z);
            ofs << ir << " " << ig << " " << ib << "\n";
        }
        ofs.close();

    } else if (subStrExt == "png") {

        unsigned char *rgba = new unsigned char[mRdrWidth*mRdrHeight*4];

        for (int i=0; i<mRdrWidth*mRdrHeight; i++) {
            int valuer = (int) ((mFrameBuffer[i].x)*255.0);
            valuer = std::max( std::min(valuer, 255), 0 );
            int valueg = (int) ((mFrameBuffer[i].y)*255.0);
            valueg = std::max( std::min(valueg, 255), 0 );
            int valueb = (int) ((mFrameBuffer[i].z)*255.0);
            valueb = std::max( std::min(valueb, 255), 0 );

            rgba[i*4 + 0] = valuer;
            rgba[i*4 + 1] = valueg;
            rgba[i*4 + 2] = valueb;
            rgba[i*4 + 3] = 255;
        }

        lodepng_encode32_file(filename, rgba, mRdrWidth, mRdrHeight);

        delete rgba;

    } else if (subStrExt == "exr") {

        EXRHeader header;
        InitEXRHeader(&header);

        header.compression_type = TINYEXR_COMPRESSIONTYPE_RLE;

        EXRImage image;
        InitEXRImage(&image);

        image.num_channels = 3;

        std::vector<float> images[3];
        for (int i = 0; i < image.num_channels; i++) images[i].resize(mRdrWidth*mRdrHeight);

        for (int i = 0; i < mRdrHeight; i++) {
          for (int j = 0; j < mRdrWidth; j++) {

              int idx = i * mRdrWidth + j;

              images[0][idx] = mFrameBuffer[idx].x;
              images[1][idx] = mFrameBuffer[idx].y;
              images[2][idx] = mFrameBuffer[idx].z;
          }
        }

        float* image_ptr[3];

        image_ptr[0] = &(images[2].at(0)); // B
        image_ptr[1] = &(images[1].at(0)); // G
        image_ptr[2] = &(images[0].at(0)); // R

        image.images = (unsigned char**)image_ptr;
        image.width = mRdrWidth;
        image.height = mRdrHeight;

        header.num_channels = 3;
        header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
        // Must be (A)BGR order, since most of EXR viewers expect this channel order.
        strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
        strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
        strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';

        header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
        header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
        for (int i = 0; i < header.num_channels; i++) {
          header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
          // for some reason HALF does not work here, so we use FLOAT
          header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of output image to be stored in .EXR
        }

        const char* err = NULL; // or nullptr in C++11 or later.
        int ret = SaveEXRImageToFile(&image, &header, filename, &err);
        assert(ret == TINYEXR_SUCCESS);

        free(header.channels);
        free(header.pixel_types);
        free(header.requested_pixel_types);

    }

    mTmr->writeToDisk_out = clock();
}
