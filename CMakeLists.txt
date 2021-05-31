cmake_minimum_required(VERSION 3.0.0)

project(match)
set(CMAKE_BUILD_TYPE RELEASE)


include_directories(src)

# find opencv
set(OpenCV_DIR /usr/local/lib/opencv3/share/OpenCV)
find_package(OpenCV 3.4.0)
include_directories(${OpenCV_INCLUDE_DIRS})




file(GLOB SOURCES src/*.cpp main.cpp)
add_executable(main ${SOURCES})
target_link_libraries(main ${OpenCV_LIBS})
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
