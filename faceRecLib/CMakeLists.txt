
cmake_minimum_required(VERSION 2.8) 
project(mylib)
FIND_PACKAGE(OpenCV 4.1 REQUIRED )
set(CMAKE_CXX_STANDARD 11)
#-------https://blog.csdn.net/zymill/article/details/81664233-------
set(CMAKE_CXX_FLAGS "-fPIC ${CMAKE_CXX_FLAGS}")
#-------------------------------------------------------------------
INCLUDE(/home/chenzl/chenzl/dlib/dlib/cmake)
#--------------------
include(/home/chenzl/chenzl/dlib/dlib/cmake_utils/use_cpp_11.cmake)
option(CUDA_PROPAGATE_HOST_FLAGS "Propage C/CXX_FLAGS and friends to the host compiler via -Xcompile" OFF)
find_package(CUDA 7.5 REQUIRED)
set(CUDA_HOST_COMPILATION_CPP ON)
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-D__STRICT_ANSI__")
add_definitions(-DDLIB_USE_CUDA)
include(/home/chenzl/chenzl/dlib/dlib/cmake_utils/test_for_cudnn/find_cudnn.txt)
if (cudnn_include AND cudnn)
   include_directories(${cudnn_include})
   cuda_add_library(cudnn_test STATIC /home/chenzl/chenzl/dlib/dlib/cuda/cudnn_dlibapi.cpp ${cudnn} )
   enable_cpp11_for_target(cudnn_test)
endif()
#---------------------
#include_directories(/home/chenzl/chenzl/C++/faceRecognition_V2/faceRecLib/)
set(mylib_dir /home/chenzl/chenzl/C++/faceRecognition_V2/faceRecLib/)
add_library(faceRecognitionlib SHARED ${mylib_dir}/faceRec.cpp ${mylib_dir}/faceRec.hpp)
TARGET_LINK_LIBRARIES(faceRecognitionlib ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(faceRecognitionlib dlib)