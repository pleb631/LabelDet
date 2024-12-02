cmake_minimum_required(VERSION 3.5.0)
project(labelDet VERSION 0.1.0 LANGUAGES C CXX)

add_executable(labelDet main.cpp include/Label.cpp include/utils.cpp include/LabelDet.cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")


include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(OpenCV_DIR D:/project/c++/lib/opencv/build)


find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

include_directories(${PROJECT_NAME}
    ${OpenCV_INCLUDE_DIRS}
    include)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
  
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})