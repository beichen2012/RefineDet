project(RefineDetTest)
cmake_minimum_required(VERSION 3.2)

add_definitions(-std=c++11)
add_definitions(-DLOG_OUT)
# OpenCV 3
find_package(OpenCV 3 QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    # OpenCV 2
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
endif()
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    # list(APPEND Caffe2_DEPENDENCY_LIBS ${OpenCV_LIBS})
    message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
else()
    message(WARNING "Not find OpenCV. project maybe invalid ")
    # caffe2_update_option(USE_OPENCV OFF)
endif()

include_directories("/usr/local/include")
include_directories(${CMAKE_CURRENT_LIST_DIR}/third_party)
include_directories(${CMAKE_CURRENT_LIST_DIR}/third_party/eigen)
link_directories("/usr/local/lib")
set(SRC ${CMAKE_CURRENT_LIST_DIR}/CaffePredict.cpp 
	${CMAKE_CURRENT_LIST_DIR}/main.cpp)
add_executable(test_refinedet ${SRC})


# caffe 
list(APPEND CMAKE_PREFIX_PATH "/opt/caffe-refinedet")
find_package(Caffe)
if(Caffe_FOUND)
    message(STATUS "Found Caffe Header at " ${Caffe_INCLUDE_DIRS})
else()
    set(Caffe_INCLUDE_DIRS " ")
    message(STATUS "Did not find Caffe! the project maybe invalid!")
endif()
include_directories( ${Caffe_INCLUDE_DIRS})


target_link_libraries(test_refinedet opencv_core opencv_highgui 
opencv_imgcodecs opencv_imgproc
caffe)
