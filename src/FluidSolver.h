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


#ifndef FLUIDSOLVER_H
#define FLUIDSOLVER_H

#include <assert.h>

#include "utils/helper_cuda.h"

#include "Grid.h"
#include "Timer.h"
#include "SceneSettings.h"
#include "Parser.h"


class FluidSolver {

private:
    SceneSettings *mScn;
    Timer *mTmr;

    Grid **d_mT;
    Grid **d_mU;
    Grid **d_mV;
    Grid **d_mW;

    // declare fluid device array
    int mByteSize;
    int mByteSizeU;
    int mByteSizeV;
    int mByteSizeW;

    float *d_mTFront;
    float *d_mTBack;
    float *d_mUFront;
    float *d_mUBack;
    float *d_mVFront;
    float *d_mVBack;
    float *d_mWFront;
    float *d_mWBack;

    float *d_mRhs;
    float *d_mP;

    // declare particle arrays
    float3 *h_mPartPos;
    float3 *h_mPartVel;
    float *h_mPartPscale;
    float *h_mPartTemp;

    float3 *d_mPartPos;
    float3 *d_mPartVel;
    float *d_mPartPscale;
    float *d_mPartTemp;

    bool mSingleFrameSourceInit;
    int mParticleCount;

    int mWidth;
    int mHeight;
    int mDepth;

    float mDensity;
    float mBuoyancy;
    float mCoolingRate;
    float mGravity;
    float mVorticeConf;
    float mDrag;

    float mTurbulenceAmp;
    float mTurbulenceScale;
    float2 mTurbMaskTempRamp;
    float2 mTurbMaskVelRamp;

    float mDx;
    float mDt;
    int mMaxIter;

    void project();

public:
    FluidSolver(Timer *tmr, SceneSettings *scn);
    ~FluidSolver();
    void addSource();
    void step ();
    Grid **getTempGrid() { return d_mT; }

};

#endif
