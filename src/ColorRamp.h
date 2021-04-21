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


#ifndef COLORRAMP_H
#define COLORRAMP_H

#include "utils/helper_math.h"

#include "SceneSettings.h"


class ColorRamp {

private:
    const int mSize;
    float *mPositions;
    float3 *mColors;
    const float mIn;
    const float mOut;

public:
    __device__ ColorRamp(SceneSettings scn)
    : mSize(scn.shaderTempRampSize), mIn(scn.shaderTempRange.x), mOut(scn.shaderTempRange.y)
    {
        mPositions = new float[mSize];
        mColors = new float3[mSize];

        for (int i=0; i<mSize; i++) {
            mPositions[i] = scn.shaderColorRampKeys[i];
            mColors[i] = scn.shaderColorRampColors[i];
        }
    }

    __device__ ~ColorRamp() {}

    inline __device__ float remap(const float &hitPos) const {
        return fminf(fmaxf((hitPos-mIn) / (mOut-mIn), 0.0f), 1.0f);
    }

    __device__ float3 getColor(const float &temp) const {

        const float remapTemp = remap(temp);

        if (mSize == 1) return mColors[0];

        int prevIdx = 0;
        int nextIdx = 0;

        for (int i=0; i<mSize; i++) {
            if (mPositions[i] <= remapTemp) {
                prevIdx = i;
            } else {
                nextIdx = i;
                break;
            }
        }

        const float range = mPositions[nextIdx] - mPositions[prevIdx];
        const float blend = (remapTemp - mPositions[prevIdx]) / range;

        return lerp(mColors[prevIdx], mColors[nextIdx], blend);
    }
};

#endif
