#include "renderer.h"

namespace l2s {

std::string compileDirectory() {
    static std::string dir = COMPILE_DIR;
    return dir;
}

} // namespace l2s
