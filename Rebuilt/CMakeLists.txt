cmake_minimum_required(VERSION 3.18)
project(sn_accumulate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)

# OpenEXR package names/targets vary between v2, v3, distro packages, and vcpkg.
find_package(OpenEXR CONFIG QUIET)
if(NOT OpenEXR_FOUND)
  find_package(OpenEXR QUIET)
endif()

add_executable(sn_accumulate sn_accumulate.cpp)

if(TARGET OpenEXR::OpenEXR)
  target_link_libraries(sn_accumulate PRIVATE OpenEXR::OpenEXR)
elseif(TARGET OpenEXR::IlmImf)
  target_link_libraries(sn_accumulate PRIVATE OpenEXR::IlmImf)
elseif(TARGET IlmImf::IlmImf)
  target_link_libraries(sn_accumulate PRIVATE IlmImf::IlmImf)
elseif(OpenEXR_FOUND)
  target_include_directories(sn_accumulate PRIVATE ${OpenEXR_INCLUDE_DIRS})
  target_link_libraries(sn_accumulate PRIVATE ${OpenEXR_LIBRARIES})
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(OPENEXR REQUIRED OpenEXR)
  target_include_directories(sn_accumulate PRIVATE ${OPENEXR_INCLUDE_DIRS})
  target_link_directories(sn_accumulate PRIVATE ${OPENEXR_LIBRARY_DIRS})
  target_link_libraries(sn_accumulate PRIVATE ${OPENEXR_LIBRARIES})
  target_compile_options(sn_accumulate PRIVATE ${OPENEXR_CFLAGS_OTHER})
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(sn_accumulate PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
  target_compile_options(sn_accumulate PRIVATE /O2 /EHsc)
else()
  target_compile_options(sn_accumulate PRIVATE -O3 -march=native)
endif()
