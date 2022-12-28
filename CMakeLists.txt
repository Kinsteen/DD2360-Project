#---------------------------------------------------------------------------------------------------
# CMake Build Configuration for the Ray Tracing Weekend Series
#
# See README.md for guidance.
#---------------------------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.1.0 )

project ( RTWeekend
  VERSION 3.0.0
  LANGUAGES CUDA CXX
)

find_package(SFML COMPONENTS 
    system window graphics network audio REQUIRED)

# Set to c++11
set ( CMAKE_CXX_STANDARD 11 )

# Set CUDA architecture. Change according to what is supported on your GPU
set ( CMAKE_CUDA_ARCHITECTURES 86 )

set ( CMAKE_BUILD_TYPE Debug )

add_compile_options ( -lineinfo -rdynamic -O3 -maxrregcount=64 )

# Source
set ( COMMON_ALL
  src/common/rtweekend.h
  src/common/camera.h
  src/common/ray.h
  src/common/vec3.h
  src/common/cuda_utils.h
  src/common/color.h
)

set ( SOURCE_ONE_WEEKEND
  ${COMMON_ALL}
  src/InOneWeekend/hittable.h
  src/InOneWeekend/hittable_list.h
  src/InOneWeekend/material.h
  src/InOneWeekend/sphere.h
  src/InOneWeekend/render.h
  src/InOneWeekend/render_gpu.h
)

# Executables
add_executable(inOneWeekend      ${SOURCE_ONE_WEEKEND} src/InOneWeekend/main.cu)
# add_executable(test      ${SOURCE_ONE_WEEKEND}  src/InOneWeekend/test.cu)
target_link_libraries(inOneWeekend PRIVATE sfml-graphics)
# target_link_libraries(test PRIVATE sfml-graphics)

include_directories(src/common)
