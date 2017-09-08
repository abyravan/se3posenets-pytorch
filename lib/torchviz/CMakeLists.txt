cmake_minimum_required(VERSION 2.8.3)
project(torchviz)

# build release if not explicitly told to build debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# turn on c++11
add_definitions(-std=c++11)

###########################################################################
# DART stuff

# set CMakeModule path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

# Boost stuff
find_package(Boost REQUIRED COMPONENTS system thread program_options)

# find packages with modules
find_package(CUDA REQUIRED)
find_package(PNG REQUIRED)
find_package(OpenGL REQUIRED QUIET)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(DEPTHSENSE)
find_package(OpenNI2)
find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRED)

# find packages with pkg-config
find_package(PkgConfig)
pkg_check_modules(eigen3 REQUIRED eigen3)
pkg_check_modules(tinyxml REQUIRED tinyxml)
pkg_check_modules(matheval REQUIRED libmatheval)
pkg_check_modules(assimp assimp)

# UtilFuncs
# SET cmake module path to find utilfuncs - for this to work,
# We need the environment variable "UTILFUNCS_ROOT_DIR" to be set to the directory where the package is present
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{UTILFUNCS_ROOT_DIR}/cmake/Modules/")
find_package(UtilFuncs REQUIRED)
IF (UTILFUNCS_FOUND)
   message(STATUS "******* Found package 'UtilFuncs' at $ENV{UTILFUNCS_ROOT_DIR} ******")
ENDIF()

message(STATUS "UtilFuncs_INCLUDE_DIRS = ${UtilFuncs_INCLUDE_DIRS}")
message(STATUS "UtilFuncs_LIBRARY_DIRS = ${UtilFuncs_LIBRARY_DIRS}")
message(STATUS "UtilFuncs_LIBRARIES = ${UtilFuncs_LIBRARIES}")

# Include dirs
include_directories(
    ${Boost_INCLUDE_DIRS}
    ${eigen3_INCLUDE_DIRS}
    ${PNG_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${GLUT_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${tinyxml_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${UtilFuncs_INCLUDE_DIRS}
)
link_directories(${tinyxml_LIBRARY_DIRS}
                 ${Pangolin_LIBRARY_DIRS}
                 ${OpenCV_LIBRARY_DIRS}
                 ${UtilFuncs_LIBRARY_DIRS}
)
link_libraries(
    ${PNG_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLUT_glut_LIBRARY}
    ${GLEW_LIBRARIES}
    ${tinyxml_LIBRARIES}
    ${matheval_LIBRARIES}
    ${Pangolin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${UtilFuncs_LIBRARIES}
)

# set dart source
set(dart_src
    src/tracker.h
    src/tracker.cpp

    src/geometry/distance_transforms.h
    src/geometry/geometry.h
    src/geometry/geometry.cpp
    src/geometry/grid_2d.h
    src/geometry/grid_3d.h
    src/geometry/sdf.h
    src/geometry/sdf.cpp
    src/geometry/SE3.h
    src/geometry/plane_fitting.h
    src/geometry/plane_fitting.cpp

    src/mesh/mesh.h
    src/mesh/mesh.cpp
    src/mesh/mesh_proc.h
    src/mesh/mesh_proc.cpp
    src/mesh/mesh_splat.h
    src/mesh/mesh_splat.cpp
    src/mesh/mesh_sample.h
    src/mesh/mesh_sample.cpp
    src/mesh/primitive_meshing.h
    src/mesh/primitive_meshing.cpp

    src/img_proc/bilateral_filter.h
    src/img_proc/img_ops.h
    src/img_proc/organized_point_cloud.h
    src/img_proc/resampling.h

    src/model/host_only_model.h
    src/model/host_only_model.cpp
    src/model/mirrored_model.h
    src/model/mirrored_model.cpp
    src/model/model.h
    src/model/model.cpp

    src/optimization/kernels/obsToMod.h
    src/optimization/kernels/modToObs.h
    src/optimization/kernels/intersection.h
    src/optimization/kernels/raycast.h
    src/optimization/contact_prior.cpp
    src/optimization/optimization.h
    src/optimization/optimizer.h
    src/optimization/optimizer.cpp
    src/optimization/point_2d_3d_prior.cpp
    src/optimization/point_3d_3d_prior.cpp
    src/optimization/prediction_renderer.h
    src/optimization/prediction_renderer.cpp
    src/optimization/priors.h

    src/pose/pose.h
    src/pose/pose.cpp
    src/pose/pose_reduction.h

    src/util/cuda_utils.h
    src/util/dart_io.h
    src/util/dart_io.cpp
    src/util/dart_types.h
    src/util/gl_dart.h
    src/util/gl_dart.cpp
    src/util/image_io.h
    src/util/image_io.cpp
    src/util/mirrored_memory.h
    src/util/model_renderer.h
    src/util/model_renderer.cpp
    src/util/ostream_operators.h
    src/util/ostream_operators.cpp
    src/util/prefix.h
    src/util/string_format.h
    src/util/string_format.cpp
    src/util/vector_type_template.h

    src/visualization/color_ramps.h
    src/visualization/data_association_viz.h
    src/visualization/gradient_viz.h
    src/visualization/matrix_viz.h
    src/visualization/sdf_viz.h
)

set(cpu_specific_src
    src/geometry/distance_transforms.cpp
    src/img_proc/img_ops.cpp
    src/img_proc/organized_point_cloud.cpp
    src/img_proc/resampling.cpp

    src/visualization/color_ramps.cpp
    src/visualization/data_association_viz.cpp
    src/visualization/gradient_viz.cpp
    src/visualization/sdf_viz.cpp
)

set(gpu_specific_src
    src/geometry/distance_transforms.cu
    src/geometry/plane_fitting.cu

    src/img_proc/bilateral_filter.cu
    src/img_proc/img_ops.cu
    src/img_proc/organized_point_cloud.cu
    src/img_proc/resampling.cu

    src/optimization/kernels/kernel_common.h
    src/optimization/kernels/obsToMod.cu
    src/optimization/kernels/modToObs.cu
    src/optimization/kernels/intersection.cu
    src/optimization/kernels/raycast.cu

    src/visualization/color_ramps.cu
    src/visualization/data_association_viz.cu
    src/visualization/gradient_viz.cu
    src/visualization/matrix_viz.cu
    src/visualization/sdf_viz.cu
)

set(point_cloud_srcs
    src/point_cloud_src/point_cloud_src.h
)

set(depth_source_src
    src/depth_sources/depth_source.h
    src/depth_sources/image_depth_source.h
)

# set conditional sources
if(OPENNI2_FOUND)
    include_directories(${OPENNI2_INCLUDE_DIRS})
    link_directories(${OPENNI2_REDIST_DIR})
    link_libraries(${OPENNI2_LIBRARY})
    set(depth_source_src
        ${depth_source_src}
        src/depth_sources/openni_depth_source.h
        src/depth_sources/openni_depth_source.cpp
    )
endif()

if(DEPTHSENSE_FOUND)
    add_definitions(-DHAS_INTEL)
    include_directories(${DEPTHSENSE_INCLUDE_DIR})
    link_libraries(${DEPTHSENSE_LIBRARIES})
    set(depth_source_src
        ${depth_source_src}
        src/depth_sources/depthsense_depth_source.h
        src/depth_sources/depthsense_depth_source.cpp
    )
endif()

if(PMD_FOUND)
    set(depth_source_src
        ${depth_source_src}
        src/depth_sources/pmd_depth_source.h
        src/depth_sources/pmd_depth_source.cpp
    )
endif()

if(assimp_FOUND)
    link_libraries(${assimp_LIBRARIES})
    include_directories(${assimp_INCLUDE_DIRS})
    set(dart_src
        ${dart_src}
        src/mesh/assimp_mesh_reader.h
        src/mesh/assimp_mesh_reader.cpp
    )
    add_definitions(-DASSIMP_BUILD)
endif()

include_directories(src/)

# set up library output
set(BUILD_SHARED_LIBS ON)

# CUDA flags
set(CUDA_NVCC_FLAGS "-arch=sm_35" "--use_fast_math"  "-O3" "--ptxas-options=--verbose") # "-fmad=false" "-DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP"

# add library
if(CUDA_FOUND)
    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
    add_definitions(-DCUDA_BUILD)
    cuda_add_library(dart ${dart_src} ${point_cloud_srcs} ${depth_source_src} ${gpu_specific_src}) # Changed ${PROJECT_NAME} to dart
else()
    add_library(dart ${dart_src} ${point_cloud_srcs} ${depth_source_src} ${cpu_specific_src}) # Changed ${PROJECT_NAME} to dart
endif()

# testing
find_package(GTest)
set(BUILD_TESTS OFF CACHE BOOL "build the tests (requires gtest)")
if(${BUILD_TESTS} AND ${GTEST_FOUND})
    message(STATUS "building tests")

    find_package(Threads REQUIRED QUIET)

    include_directories(${GTEST_INCLUDE_DIRS})
    link_directories(${GTEST_LIBRARIES})
    link_libraries(gtest pthread)

    set(test_sources
        test/test_model_jacobian.cpp
        test/test_obs_to_mod_kernels.cpp
        test/test_pose_reduction.cpp
        test/test_voxelize.cpp
    )

    set(gpu_test_sources
        test/test_model_jacobian.cu
        test/test_mirrored_model.cpp
    )

    cuda_add_executable(
                # executable
                test
                # main source
                test/test_main.cpp
                # test sources
                ${test_sources}
                # other sources
                ${sources}
                ${gpu_sources}
                ${gpu_test_sources}
    )
    target_link_libraries(test dart)
endif()

##########################
# PYTORCH VISUALIZATIONS

# For new renderer
set(compileDirectory \"${PROJECT_SOURCE_DIR}\")
add_definitions(-DCOMPILE_DIR=${compileDirectory})

# Visualizer
add_library(pangoviz
             pangolin_visualizer.cpp
             src/render/renderer.cpp)
target_link_libraries(pangoviz dart ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${UtilFuncs_LIBRARIES})

# Visualizer
add_library(pangoposeviz
             pose_visualizer.cpp
             src/render/renderer.cpp)
target_link_libraries(pangoposeviz dart ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${UtilFuncs_LIBRARIES})

# Data Visualizer
add_library(pangodataviz
             data_visualizer.cpp
             src/render/renderer.cpp)
target_link_libraries(pangodataviz dart ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${UtilFuncs_LIBRARIES})
