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


#ifndef RAY_H
#define RAY_H


class Ray {

public:
    const float3 mOrig;
    const float3 mDir;
    const float3 mInvdir;
    const uint3 mSign;
    float mT0, mT1;

    __device__ Ray(const float3 &orig, const float3 &dir)
        : mOrig(orig), mDir(dir), mInvdir(1.0f / dir), mT0(0.0f), mT1(uint_as_float(0x7f800000)),
          mSign(make_uint3( mInvdir.x < 0.0f, mInvdir.y < 0.0f, mInvdir.z < 0.0f ))
    { }

    __device__ inline float3 at(float t) const {
        return mOrig + t*mDir;
    }

};

#endif
