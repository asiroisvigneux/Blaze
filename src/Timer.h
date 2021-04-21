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


#ifndef TIMER_H
#define TIMER_H

#include <cstdio>
#include <time.h>


class Timer {

public:
    int iter;
    float time;

    clock_t source_in;
    clock_t source_out;

    clock_t cooldown_in;
    clock_t cooldown_out;

    clock_t drag_in;
    clock_t drag_out;

    clock_t buoyancy_in;
    clock_t buoyancy_out;

    clock_t vorticity_in;
    clock_t vorticity_out;

    clock_t wind_in;
    clock_t wind_out;

    clock_t turbulence_in;
    clock_t turbulence_out;

    clock_t computeDivergence_in;
    clock_t computeDivergence_out;

    clock_t gsSolve_in;
    clock_t gsSolve_out;

    clock_t pressureGradientUpdate_in;
    clock_t pressureGradientUpdate_out;

    clock_t advect_in;
    clock_t advect_out;

    clock_t scatter_in;
    clock_t scatter_out;

    clock_t render_in;
    clock_t render_out;

    clock_t writeToDisk_in;
    clock_t writeToDisk_out;

    std::string getStats() {
        float sourceTime = (double)((double)(source_out - source_in)/CLOCKS_PER_SEC);
        float cooldownTime = (double)((double)(cooldown_out - cooldown_in)/CLOCKS_PER_SEC);
        float dragTime = (double)((double)(drag_out - drag_in)/CLOCKS_PER_SEC);
        float buoyancyTime = (double)((double)(buoyancy_out - buoyancy_in)/CLOCKS_PER_SEC);
        float vorticityTime = (double)((double)(vorticity_out - vorticity_in)/CLOCKS_PER_SEC);
        float windTime = (double)((double)(wind_out - wind_in)/CLOCKS_PER_SEC);
        float turbulenceTime = (double)((double)(turbulence_out - turbulence_in)/CLOCKS_PER_SEC);
        float computeDivergenceTime = (double)((double)(computeDivergence_out - computeDivergence_in)/CLOCKS_PER_SEC);
        float gsSolveTime = (double)((double)(gsSolve_out - gsSolve_in)/CLOCKS_PER_SEC);
        float applyPressureGradientTime = (double)((double)(pressureGradientUpdate_out - pressureGradientUpdate_in)/CLOCKS_PER_SEC);
        float advectTime = (double)((double)(advect_out - advect_in)/CLOCKS_PER_SEC);
        float scatterTime = (double)((double)(scatter_out - scatter_in)/CLOCKS_PER_SEC);
        float renderTime = (double)((double)(render_out - render_in)/CLOCKS_PER_SEC);
        float writeToDiskTime = (double)((double)(writeToDisk_out - writeToDisk_in)/CLOCKS_PER_SEC);

        float totalTime = sourceTime
                        + cooldownTime
                        + dragTime
                        + buoyancyTime
                        + vorticityTime
                        + windTime
                        + turbulenceTime
                        + computeDivergenceTime
                        + gsSolveTime
                        + applyPressureGradientTime
                        + advectTime
                        + scatterTime
                        + renderTime
                        + writeToDiskTime;

        std::string msg = "";
        char tmp[256];

        sprintf(tmp, "\nFrame: %03d\n", iter); msg += tmp;
        sprintf(tmp, "Sourcing:                 %4.6f s,   ratio: %1.2f\n", sourceTime, sourceTime/totalTime); msg += tmp;
        sprintf(tmp, "Cooldown:                 %4.6f s,   ratio: %1.2f\n", cooldownTime, cooldownTime/totalTime); msg += tmp;
        sprintf(tmp, "Drag:                     %4.6f s,   ratio: %1.2f\n", dragTime, dragTime/totalTime); msg += tmp;
        sprintf(tmp, "Buoyancy:                 %4.6f s,   ratio: %1.2f\n", buoyancyTime, buoyancyTime/totalTime); msg += tmp;
        sprintf(tmp, "Vorticity Confinement:    %4.6f s,   ratio: %1.2f\n", vorticityTime, vorticityTime/totalTime); msg += tmp;
        sprintf(tmp, "Wind:                     %4.6f s,   ratio: %1.2f\n", windTime, windTime/totalTime); msg += tmp;
        sprintf(tmp, "Turbulence:               %4.6f s,   ratio: %1.2f\n", turbulenceTime, turbulenceTime/totalTime); msg += tmp;
        sprintf(tmp, "Compute Divergence:       %4.6f s,   ratio: %1.2f\n", computeDivergenceTime, computeDivergenceTime/totalTime); msg += tmp;
        sprintf(tmp, "Gauss-Seidel Solve:       %4.6f s,   ratio: %1.2f\n", gsSolveTime, gsSolveTime/totalTime); msg += tmp;
        sprintf(tmp, "Apply Pressure Gradient:  %4.6f s,   ratio: %1.2f\n", applyPressureGradientTime, applyPressureGradientTime/totalTime); msg += tmp;
        sprintf(tmp, "Advection:                %4.6f s,   ratio: %1.2f\n", advectTime, advectTime/totalTime); msg += tmp;
        sprintf(tmp, "Scattering:               %4.6f s,   ratio: %1.2f\n", scatterTime, scatterTime/totalTime); msg += tmp;
        sprintf(tmp, "Render:                   %4.6f s,   ratio: %1.2f\n", renderTime, renderTime/totalTime); msg += tmp;
        sprintf(tmp, "Write to Disk:            %4.6f s,   ratio: %1.2f\n", writeToDiskTime, writeToDiskTime/totalTime); msg += tmp;
        sprintf(tmp, "Total Time:               %4.6f s,   ratio: %1.2f\n", totalTime, 1.0f); msg += tmp;

        return msg;
    }

};

#endif
