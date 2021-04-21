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


#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

class Logger{

private:
    std::string mLogFile;

    // https://stackoverflow.com/questions/7400418/writing-a-log-file-in-c-c
    inline std::string getCurrentDateTime( std::string s ){
        time_t now = time(0);
        struct tm  tstruct;
        char  buf[80];
        tstruct = *localtime(&now);
        if(s=="now")
            strftime(buf, sizeof(buf), "%Y-%m-%d_%X", &tstruct);
        else if(s=="date")
            strftime(buf, sizeof(buf), "%Y-%m-%d", &tstruct);
        return std::string(buf);
    };

public:
    Logger(std::string sceneDir, std::string sceneName) {
        std::string logDir(sceneDir + "/../logs/");
        mLogFile = logDir + "log_" + sceneName + "_" + getCurrentDateTime("now") + ".txt";

        mkdir(logDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }

    void append(std::string logMsg){
        std::ofstream ofs(mLogFile, std::ios_base::out | std::ios_base::app );
        ofs << logMsg;
        ofs.close();
    }

};

#endif
