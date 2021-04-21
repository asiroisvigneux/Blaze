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


#ifndef SHADER_H
#define SHADER_H

#include "SceneSettings.h"
#include "ColorRamp.h"


class Shader {

public:
    float mDensityScale;
    float mShadowDensityScale;

    float3 mVolumeColor;
    float3 mScattering;
    float mAbsorption;
    float mSGain;

    float3 mAlbedo;
    float3 mExtinction;

    float mEmissionScale;
    ColorRamp *mEmissionColorRamp;

    float mMultiScatterScale;
    float2 mMultiScatterDensityMask;
    float3 mMultiScatterColor;

    __device__ Shader(SceneSettings scn)
      : mDensityScale(scn.shaderDensityScale)
      , mShadowDensityScale(scn.shaderShadowDensityScale)
      , mAbsorption(scn.shaderAbsorption)
      , mScattering(scn.shaderScattering)
      , mVolumeColor(scn.shaderVolumeColor)
      , mAlbedo((mVolumeColor*mScattering) / (mScattering + make_float3(mAbsorption)))
      , mSGain(scn.shaderGain)
      , mExtinction((mScattering+mAbsorption)*-1.0f)
      , mEmissionColorRamp(new ColorRamp(scn))
      , mEmissionScale(scn.shaderEmissionScale)
      , mMultiScatterScale(scn.multiScatterScale)
      , mMultiScatterDensityMask(scn.multiScatterDensityMask)
      , mMultiScatterColor(scn.multiScatterColor)
    {}

    __device__ ~Shader(){
        delete mEmissionColorRamp;
    }

    __device__ float3 getEmissionColor(float temp) {
        return mEmissionColorRamp->getColor(temp);
    }

};

#endif
