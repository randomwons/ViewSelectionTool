#include "common.h"
#include <fstream>
#include <sstream>

std::optional<std::string> LoadTextFile(const std::string& filename){
    std::ifstream fin(filename);
    if(!fin.is_open()){
        printf("Failed to open file '%s'\n", filename.c_str());
        return {};
    }
    std::stringstream text;
    text << fin.rdbuf();
    return text.str();
}