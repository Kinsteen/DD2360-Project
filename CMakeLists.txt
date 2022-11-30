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

# Set to c++11
set ( CMAKE_CXX_STANDARD 11 )

# Set CUDA architecture. Change according to what is supported on your GPU
set ( CMAKE_CUDA_ARCHITECTURES 86 )

# Source
set ( COMMON_ALL
  src/common/rtweekend.h
  src/common/camera.h
  src/common/ray.h
  src/common/vec3.h
)

set ( SOURCE_ONE_WEEKEND
  ${COMMON_ALL}
  src/InOneWeekend/hittable.h
  src/InOneWeekend/hittable_list.h
  src/InOneWeekend/material.h
  src/InOneWeekend/sphere.h
  src/InOneWeekend/main.cu
)

# Executables
add_executable(inOneWeekend      ${SOURCE_ONE_WEEKEND})

include_directories(src/common)