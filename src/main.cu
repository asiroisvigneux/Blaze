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


#include "SceneSettings.h"
#include "Parser.h"
#include "Logger.h"
#include "FluidSolver.h"
#include "RenderEngine.h"


void printHelp() {
    printf("blaze version 0.1.0\n"
            "Usage: blaze [options] <blz scene file>\n"
            " -l             : log runtime performance data to a txt file\n"
            " -h             : print usage information\n"
            " -help          : print usage information\n"
            " --help         : print usage information\n");
}

bool argParse(int argc, char* argv[], char* scenePath, bool &writeLog) {
    if (argc == 1 || (argc == 2 && strcmp(argv[1], "-l") == 0)) {
        printf("No scene file provided, blaze will be terminated\n");
        return false;
    } else if (argc > 3 || (argc == 3 && strcmp(argv[1], "-l") != 0)) {
        printf("Only a single scene file should be provided, blaze will be terminated\n");
        return false;
    } else {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "-help") == 0 || strcmp(argv[1], "--help") == 0) {
            printHelp();
            return false;
        }
        if (strcmp(argv[1], "-l") == 0) {
            writeLog = true;
        }
        std::string fn = argv[writeLog+1];
        if(fn.substr(fn.find_last_of(".") + 1) != "blz") {
            printf("The provided scene file should have a .blz extention, blaze will be terminated\n");
            return false;
        }
        std::ifstream f(argv[writeLog+1]);
        if (!f.good()) {
            printf("Scene file %s does not exist on disk\n", argv[writeLog+1]);
            return false;
        }
    }

    strcpy(scenePath, argv[writeLog+1]);
    return true;
}

int main(int argc, char* argv[]) {

    char sceneFilePath[256];
    bool writeLog = false;
    if (!argParse(argc, argv, sceneFilePath, writeLog)) return 1;

    SceneSettings scn;
    Parser::sceneParse(&scn, sceneFilePath);
    Logger *log;
    if(writeLog) {
        log = new Logger(scn.sceneDir, scn.sceneName);
        log->append(scn.getSceneInfo());
    }
    Timer *tmr = new Timer();
    RenderEngine *renderer = new RenderEngine(tmr, scn);
    FluidSolver *solver = new FluidSolver(tmr, &scn);

    tmr->time = 0.0f;
    tmr->iter = scn.sourceRange.x;
    while(tmr->iter <= scn.renderRange.y) {
        if (tmr->iter <= scn.sourceRange.y) {
            solver->addSource();
        }
        solver->step();
        if (tmr->iter >= scn.renderRange.x && tmr->iter <= scn.renderRange.y) {
            renderer->render(solver->getTempGrid());
            renderer->writeToDisk();
        }
        std::string msg(tmr->getStats());
        printf(msg.c_str());
        if(writeLog)
            log->append(msg);
        tmr->iter++;
        tmr->time += scn.dt;
    }

    delete renderer;
    delete solver;
    delete tmr;
    if(writeLog)
        delete log;

    cudaDeviceReset();
    return 0;

}
