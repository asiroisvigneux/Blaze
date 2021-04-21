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


#ifndef RENDERENGINE_H
#define RENDERENGINE_H

#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>

#include "utils/helper_cuda.h"

#include "SceneSettings.h"
#include "Shader.h"
#include "Camera.h"
#include "Volume.h"
#include "Ray.h"
#include "Light.h"
#include "Timer.h"
#include "Grid.h"
#include "Parser.h"


class RenderEngine {

private:
    Timer *mTmr;

    Shader **d_mShader;
    Camera **d_mCamera;
    Light **d_mLights;
    int mLightCount;

    int mWidth;
    int mHeight;
    int mDepth;
    float mDx;

    float mPStep;
    float mSStep;
    float mCutoff;

    int mRdrWidth;
    int mRdrHeight;

    std::string mRenderFile;
    std::string mSceneDir;

    float3 *mFrameBuffer;

    Volume *mTempVol;
    Volume *mScatterFrontVol;
    Volume *mScatterBackVol;
    static const int mScatterResRatio = 4;

    static const int mFilterWidth = 3;
    static const int mFilterSize = mFilterWidth*mFilterWidth*mFilterWidth;
    static const float mFilterBlur[mFilterSize];

    float mScatterScale;
    float mScatterTempMin;
    int mScatterBlurIter;

public:
    RenderEngine(Timer *tmr, SceneSettings rampParm);
    ~RenderEngine();

    void render(Grid **tempGrid);
    void writeToDisk();

};

#endif
