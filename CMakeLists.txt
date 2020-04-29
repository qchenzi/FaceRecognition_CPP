



cmake_minimum_required(VERSION 2.8) 
project(mytest)
FIND_PACKAGE(OpenCV 4.1 REQUIRED )
#FIND_PACKAGE(OpenCV REQUIRED )
set(CMAKE_CXX_STANDARD 11)
#-------https://blog.csdn.net/zymill/article/details/81664233-------
set(CMAKE_CXX_FLAGS "-fPIC -lstdc++ ${CMAKE_CXX_FLAGS}")
#-------------------------------------------------------------------
INCLUDE(/home/chenzl/chenzl/dlib/dlib/cmake)
#--------------------
#include(/home/chenzl/chenzl/dlib/dlib/cmake_utils/use_cpp_11.cmake)
#option(CUDA_PROPAGATE_HOST_FLAGS "Propage C/CXX_FLAGS and friends to the host compiler via -Xcompile" OFF)
#find_package(CUDA 7.5 REQUIRED)
#set(CUDA_HOST_COMPILATION_CPP ON)
#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-D__STRICT_ANSI__")
#add_definitions(-DDLIB_USE_CUDA)
#include(/home/chenzl/chenzl/dlib/dlib/cmake_utils/test_for_cudnn/find_cudnn.txt)
#if (cudnn_include AND cudnn)
#   include_directories(${cudnn_include})
#   cuda_add_library(cudnn_test STATIC /home/chenzl/chenzl/dlib/dlib/cuda/cudnn_dlibapi.cpp ${cudnn} )
#   enable_cpp11_for_target(cudnn_test)
#endif()
#---------------------


#set(SOURCE_FILES /home/chenzl/chenzl/C++/faceRecognition_V2/FR.cpp)
add_executable(FR /home/chenzl/chenzl/C++/faceRecognition_V2/FR.cpp)
TARGET_LINK_LIBRARIES(FR ${OpenCV_LIBS})
#=====add myopencv class=========
INCLUDE_DIRECTORIES(/home/chenzl/chenzl/C++/faceRecognition_V2/faceRecLib/) #where is include file
LINK_DIRECTORIES(/home/chenzl/chenzl/C++/faceRecognition_V2/faceRecLib/) #告诉链接器动态链接库在哪个位置
TARGET_LINK_LIBRARIES(FR /home/chenzl/chenzl/C++/faceRecognition_V2/faceRecLib/libfaceRecognitionlib.so)
#================================




