# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/chenzl/chenzl/C++/faceRecognition_V2/faceRecLib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chenzl/chenzl/C++/faceRecognition_V2/faceRecLib

# Include any dependencies generated for this target.
include CMakeFiles/faceRecognitionlib.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/faceRecognitionlib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/faceRecognitionlib.dir/flags.make

CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.o: CMakeFiles/faceRecognitionlib.dir/flags.make
CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.o: faceRec.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chenzl/chenzl/C++/faceRecognition_V2/faceRecLib/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.o -c /home/chenzl/chenzl/C++/faceRecognition_V2/faceRecLib/faceRec.cpp

CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chenzl/chenzl/C++/faceRecognition_V2/faceRecLib/faceRec.cpp > CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.i

CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chenzl/chenzl/C++/faceRecognition_V2/faceRecLib/faceRec.cpp -o CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.s

CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.o.requires:

.PHONY : CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.o.requires

CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.o.provides: CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.o.requires
	$(MAKE) -f CMakeFiles/faceRecognitionlib.dir/build.make CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.o.provides.build
.PHONY : CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.o.provides

CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.o.provides.build: CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.o


# Object files for target faceRecognitionlib
faceRecognitionlib_OBJECTS = \
"CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.o"

# External object files for target faceRecognitionlib
faceRecognitionlib_EXTERNAL_OBJECTS =

libfaceRecognitionlib.so: CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.o
libfaceRecognitionlib.so: CMakeFiles/faceRecognitionlib.dir/build.make
libfaceRecognitionlib.so: /usr/local/opencv-4.1.0/build/lib/libopencv_gapi.so.4.1.0
libfaceRecognitionlib.so: /usr/local/opencv-4.1.0/build/lib/libopencv_stitching.so.4.1.0
libfaceRecognitionlib.so: /usr/local/opencv-4.1.0/build/lib/libopencv_ml.so.4.1.0
libfaceRecognitionlib.so: /usr/local/opencv-4.1.0/build/lib/libopencv_dnn.so.4.1.0
libfaceRecognitionlib.so: /usr/local/opencv-4.1.0/build/lib/libopencv_objdetect.so.4.1.0
libfaceRecognitionlib.so: /usr/local/opencv-4.1.0/build/lib/libopencv_video.so.4.1.0
libfaceRecognitionlib.so: /usr/local/opencv-4.1.0/build/lib/libopencv_photo.so.4.1.0
libfaceRecognitionlib.so: dlib_build/libdlib.a
libfaceRecognitionlib.so: /usr/local/opencv-4.1.0/build/lib/libopencv_calib3d.so.4.1.0
libfaceRecognitionlib.so: /usr/local/opencv-4.1.0/build/lib/libopencv_features2d.so.4.1.0
libfaceRecognitionlib.so: /usr/local/opencv-4.1.0/build/lib/libopencv_flann.so.4.1.0
libfaceRecognitionlib.so: /usr/local/opencv-4.1.0/build/lib/libopencv_highgui.so.4.1.0
libfaceRecognitionlib.so: /usr/local/opencv-4.1.0/build/lib/libopencv_videoio.so.4.1.0
libfaceRecognitionlib.so: /usr/local/opencv-4.1.0/build/lib/libopencv_imgcodecs.so.4.1.0
libfaceRecognitionlib.so: /usr/local/opencv-4.1.0/build/lib/libopencv_imgproc.so.4.1.0
libfaceRecognitionlib.so: /usr/local/opencv-4.1.0/build/lib/libopencv_core.so.4.1.0
libfaceRecognitionlib.so: /usr/local/cuda/lib64/libcudart_static.a
libfaceRecognitionlib.so: /usr/lib/x86_64-linux-gnu/librt.so
libfaceRecognitionlib.so: /usr/lib/x86_64-linux-gnu/librt.so
libfaceRecognitionlib.so: /usr/lib/x86_64-linux-gnu/libnsl.so
libfaceRecognitionlib.so: /usr/lib/x86_64-linux-gnu/libSM.so
libfaceRecognitionlib.so: /usr/lib/x86_64-linux-gnu/libICE.so
libfaceRecognitionlib.so: /usr/lib/x86_64-linux-gnu/libX11.so
libfaceRecognitionlib.so: /usr/lib/x86_64-linux-gnu/libXext.so
libfaceRecognitionlib.so: /home/chenzl/Anaconda3/lib/libpng.so
libfaceRecognitionlib.so: /home/chenzl/Anaconda3/lib/libz.so
libfaceRecognitionlib.so: /home/chenzl/Anaconda3/lib/libjpeg.so
libfaceRecognitionlib.so: /home/chenzl/Anaconda3/lib/libmkl_rt.so
libfaceRecognitionlib.so: /usr/local/cuda/lib64/libcublas.so
libfaceRecognitionlib.so: /usr/lib/x86_64-linux-gnu/libcudnn.so
libfaceRecognitionlib.so: /usr/local/cuda/lib64/libcurand.so
libfaceRecognitionlib.so: /usr/local/cuda/lib64/libcusolver.so
libfaceRecognitionlib.so: /usr/local/cuda/lib64/libcudart.so
libfaceRecognitionlib.so: /home/chenzl/Anaconda3/lib/libiomp5.so
libfaceRecognitionlib.so: /home/chenzl/Anaconda3/lib/libsqlite3.so
libfaceRecognitionlib.so: CMakeFiles/faceRecognitionlib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chenzl/chenzl/C++/faceRecognition_V2/faceRecLib/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libfaceRecognitionlib.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/faceRecognitionlib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/faceRecognitionlib.dir/build: libfaceRecognitionlib.so

.PHONY : CMakeFiles/faceRecognitionlib.dir/build

CMakeFiles/faceRecognitionlib.dir/requires: CMakeFiles/faceRecognitionlib.dir/faceRec.cpp.o.requires

.PHONY : CMakeFiles/faceRecognitionlib.dir/requires

CMakeFiles/faceRecognitionlib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/faceRecognitionlib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/faceRecognitionlib.dir/clean

CMakeFiles/faceRecognitionlib.dir/depend:
	cd /home/chenzl/chenzl/C++/faceRecognition_V2/faceRecLib && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chenzl/chenzl/C++/faceRecognition_V2/faceRecLib /home/chenzl/chenzl/C++/faceRecognition_V2/faceRecLib /home/chenzl/chenzl/C++/faceRecognition_V2/faceRecLib /home/chenzl/chenzl/C++/faceRecognition_V2/faceRecLib /home/chenzl/chenzl/C++/faceRecognition_V2/faceRecLib/CMakeFiles/faceRecognitionlib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/faceRecognitionlib.dir/depend

