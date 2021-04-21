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


#ifndef PARSER_H
#define PARSER_H

#include <iostream>
#include <string>
#include <fstream>
#include <streambuf>

#include "../thirdparty/rapidjson/document.h"
#include "../thirdparty/rapidjson/writer.h"
#include "../thirdparty/rapidjson/stringbuffer.h"


class Parser {

public:
    static void sceneParse(SceneSettings *scene, const char *sceneFile) {

        char actualpath [256];
        realpath(sceneFile, actualpath);
        strcpy(scene->sceneDir, getDirPath(actualpath).c_str());

        std::ifstream t(sceneFile);
        std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

        const char *json = str.c_str();
        rapidjson::Document d;
        d.Parse(json);

        strcpy(scene->sceneName, d["sceneName"].GetString());

        strcpy(scene->particleFile, d["particleFile"].GetString());
        scene->sourceAnimated = d["sourceAnimated"].GetInt();
        scene->sourceRange = make_int2((int)d["sourceRange"][0].GetFloat(),
                                       (int)d["sourceRange"][1].GetFloat());
        scene->sourceMaxParticleCount = d["sourceMaxParticleCount"].GetInt();

        strcpy(scene->renderFile, d["renderFile"].GetString());
        scene->renderRange = make_int2((int)d["renderRange"][0].GetFloat(),
                                       (int)d["renderRange"][1].GetFloat());

        scene->maxIterSolve = d["maxIterSolve"].GetInt();
        scene->dx = d["dx"].GetFloat();
        scene->dt = d["dt"].GetFloat();

        scene->gridRes = make_int3(d["gridRes"][0].GetInt(),
                                   d["gridRes"][1].GetInt(),
                                   d["gridRes"][2].GetInt());
        scene->renderRes = make_int2(d["renderRes"][0].GetInt(),
                                     d["renderRes"][1].GetInt());

        scene->domainBboxMin = make_float3(d["domainBboxMin"][0].GetFloat(),
                                           d["domainBboxMin"][1].GetFloat(),
                                           d["domainBboxMin"][2].GetFloat());
        scene->domainBboxMax = make_float3(d["domainBboxMax"][0].GetFloat(),
                                           d["domainBboxMax"][1].GetFloat(),
                                           d["domainBboxMax"][2].GetFloat());
        scene->closedBounds[0] = (bool)d["closedBounds"][0].GetInt();
        scene->closedBounds[1] = (bool)d["closedBounds"][1].GetInt();
        scene->closedBounds[2] = (bool)d["closedBounds"][2].GetInt();
        scene->closedBounds[3] = (bool)d["closedBounds"][3].GetInt();
        scene->closedBounds[4] = (bool)d["closedBounds"][4].GetInt();
        scene->closedBounds[5] = (bool)d["closedBounds"][5].GetInt();

        scene->density = d["density"].GetFloat();
        scene->gravity = d["gravity"].GetFloat();
        scene->coolingRate = d["coolingRate"].GetFloat();
        scene->buoyancy = d["buoyancy"].GetFloat();
        scene->vorticityConf = d["vorticityConf"].GetFloat();
        scene->drag = d["drag"].GetFloat();

        scene->windAmp = d["windAmp"].GetFloat();
        scene->windSpeed = d["windSpeed"].GetFloat();
        scene->windTurbAmp = d["windTurbAmp"].GetFloat();
        scene->windTurbScale = d["windTurbScale"].GetFloat();
        scene->windDir = d["windDir"].GetInt();

        scene->turbulence_amp = d["turbulenceAmp"].GetFloat();
        scene->turbulence_scale = d["turbulenceScale"].GetFloat();
        scene->turbMaskTempRamp = make_float2(d["turbMaskTempRamp"][0].GetFloat(),
                                              d["turbMaskTempRamp"][1].GetFloat());
        scene->turbMaskVelRamp = make_float2(d["turbMaskVelRamp"][0].GetFloat(),
                                             d["turbMaskVelRamp"][1].GetFloat());

        scene->lightCount = d["lightCount"].GetInt();
        for (int i=0; i<scene->lightCount; i++) {
            scene->lightExposure[i] = d["lights"][i]["light_exposure"].GetFloat();
            scene->lightIntensity[i] = d["lights"][i]["light_intensity"].GetFloat();
            scene->lightSamples[i] = d["lights"][i]["light_samples"].GetInt();
            scene->lightDir[i] = make_float3(d["lights"][i]["light_dir"][0].GetFloat(),
                                             d["lights"][i]["light_dir"][1].GetFloat(),
                                             d["lights"][i]["light_dir"][2].GetFloat());
            scene->lightColor[i] = make_float3(d["lights"][i]["light_color"][0].GetFloat(),
                                               d["lights"][i]["light_color"][1].GetFloat(),
                                               d["lights"][i]["light_color"][2].GetFloat());
        }

        scene->camFocal = d["cam"]["focal"].GetFloat();
        scene->camAperture = d["cam"]["aperture"].GetFloat();
        scene->camU = make_float3(d["cam"]["rotMat"][0][0].GetFloat(),
                                  d["cam"]["rotMat"][0][1].GetFloat(),
                                  d["cam"]["rotMat"][0][2].GetFloat());
        scene->camV = make_float3(d["cam"]["rotMat"][1][0].GetFloat(),
                                  d["cam"]["rotMat"][1][1].GetFloat(),
                                  d["cam"]["rotMat"][1][2].GetFloat());
        scene->camW = make_float3(d["cam"]["rotMat"][2][0].GetFloat(),
                                  d["cam"]["rotMat"][2][1].GetFloat(),
                                  d["cam"]["rotMat"][2][2].GetFloat());
        scene->camTrans = make_float3(d["cam"]["t"][0].GetFloat(),
                                      d["cam"]["t"][1].GetFloat(),
                                      d["cam"]["t"][2].GetFloat());

        scene->rndrPStep = d["primarySampleRate"].GetFloat();
        scene->rndrSStep = d["shadowSamplingRate"].GetFloat();
        scene->rndrCutoff = d["cutoff"].GetFloat();

        scene->shaderDensityScale = d["densityScale"].GetFloat();
        scene->shaderShadowDensityScale = d["shadowDensityScale"].GetFloat();
        scene->shaderVolumeColor = make_float3(d["volumeColor"][0].GetFloat(),
                                         d["volumeColor"][1].GetFloat(),
                                         d["volumeColor"][2].GetFloat());
        scene->shaderScattering = make_float3(d["scatteringColor"][0].GetFloat(),
                                        d["scatteringColor"][1].GetFloat(),
                                        d["scatteringColor"][2].GetFloat());
        scene->shaderAbsorption = d["absorption"].GetFloat();
        scene->shaderGain = d["gain"].GetFloat();


        scene->shaderEmissionScale = d["emissionScale"].GetFloat();
        scene->shaderTempRampSize = d["shaderColorRamp"]["count"].GetInt();
        scene->shaderTempRange = make_float2(d["shaderTempRange"][0].GetFloat(),
                                             d["shaderTempRange"][1].GetFloat());
        for (int i=0; i<scene->shaderTempRampSize; i++) {
            scene->shaderColorRampKeys[i] = d["shaderColorRamp"]["keys"][i].GetFloat();
            scene->shaderColorRampColors[i] = make_float3(d["shaderColorRamp"]["colors"][i][0].GetFloat(),
                                                          d["shaderColorRamp"]["colors"][i][1].GetFloat(),
                                                          d["shaderColorRamp"]["colors"][i][2].GetFloat());
        }

        scene->multiScatterScale = d["multiScatterScale"].GetFloat();
        scene->multiScatterDensityMask = make_float2( d["multiScatterDensityMask"][0].GetFloat(),
                                                      d["multiScatterDensityMask"][1].GetFloat() );
        scene->multiScatterColor = make_float3( d["multiScatterColor"][0].GetFloat(),
                                                d["multiScatterColor"][1].GetFloat(),
                                                d["multiScatterColor"][2].GetFloat() );
        scene->multiScatterBlurIter = d["multiScatterBlurIter"].GetInt();

    }

    static void sourceParticleParse(const SceneSettings *scn, float3 *posArray,
                                    float3 *velArray, float *pscaleArray, float *tempArray,
                                    int &pointCount, int frameNum)
    {
        char particleFile[256];


        // replace the frame number for animated source;
        if (scn->sourceAnimated) {
            std::string stringParticleFile(scn->particleFile);
            std::string subStr = stringParticleFile.substr(0, stringParticleFile.length()-10);
            sprintf(particleFile, "%s/%s_%04d.part", scn->sceneDir, subStr.c_str(), frameNum);
        } else {
            sprintf(particleFile, "%s/%s", scn->sceneDir, scn->particleFile);
        }

        // check if file exist
        std::ifstream f(particleFile);
        if (!f.good())
            printf("Particle file %s does not exist on disk\n", particleFile);
        assert(f.good());

        float array[8];
        std::ifstream fin(particleFile, std::ios::binary);
        int i = 0;
        while (fin.read(reinterpret_cast<char*>(&array[0]), 8*sizeof(float))) {
            posArray[i] = make_float3(array[0],
                                      array[1],
                                      array[2]);
            velArray[i] = make_float3(array[3],
                                      array[4],
                                      array[5]);
            pscaleArray[i] = array[6];
            tempArray[i] = array[7];
            i++;
        }
        pointCount = i;
    }

    static std::string getDirPath(const std::string& fname)
    {
         size_t pos = fname.find_last_of("\\/");
         return (std::string::npos == pos)
             ? ""
             : fname.substr(0, pos);
    }

};


#endif
