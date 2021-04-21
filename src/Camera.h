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


#ifndef CAMERA_H
#define CAMERA_H

#include "Ray.h"


class Camera {

public:
    const float3 mOrigin;
    const float3 mU, mV, mW;
    const uint2 mRes;
    float3 mTopLeftCorner;
    float3 mHorizontal;
    float3 mVertical;
    float mPixelWidth;

    __device__ Camera(float3 origin, float3 u, float3 v, float3 w, float focal, float aperture, int xres, int yres)
        : mOrigin(origin), mU(u), mV(v), mW(w), mRes(make_uint2(xres, yres))
    {
        float aspect = (float)xres / (float)yres;
        float halfWidth = 0.5f*aperture;
        float halfHeight = halfWidth / aspect;

        mPixelWidth = halfWidth / (xres/2.0f);
        mTopLeftCorner = origin - focal*mW - halfWidth*mU + halfHeight*mV;
        mHorizontal = 2.0f*halfWidth*u;
        mVertical = -2.0f*halfHeight*v;
    }

    __device__ Ray getRay(int i, int j) {

        return Ray(mOrigin, normalize(mTopLeftCorner + ((i+0.5f)/mRes.x)*mHorizontal
                                                     + ((j+0.5f)/mRes.y)*mVertical
                                                     - mOrigin));
    }

};

#endif
