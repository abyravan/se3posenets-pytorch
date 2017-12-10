# Find Pangolin (https://github.com/stevenlovegrove/Pangolin)
find_package(Pangolin 0.4 REQUIRED)
find_package(OpenCV)
find_package(Eigen)
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Eigen_INCLUDE_DIRS})

add_definitions(-std=c++11)

add_executable(displaynormals main.cpp)
target_link_libraries(displaynormals ${Pangolin_LIBRARIES} ${OpenCV_LIBRARIES})
