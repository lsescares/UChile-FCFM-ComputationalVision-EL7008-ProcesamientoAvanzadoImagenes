# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/luis/Escritorio/tarea2/parte1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/luis/Escritorio/tarea2/parte1/build

# Include any dependencies generated for this target.
include CMakeFiles/tarea2_parte1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tarea2_parte1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tarea2_parte1.dir/flags.make

CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.o: CMakeFiles/tarea2_parte1.dir/flags.make
CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.o: ../main_parte1.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luis/Escritorio/tarea2/parte1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.o -c /home/luis/Escritorio/tarea2/parte1/main_parte1.cpp

CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/luis/Escritorio/tarea2/parte1/main_parte1.cpp > CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.i

CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/luis/Escritorio/tarea2/parte1/main_parte1.cpp -o CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.s

CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.o.requires:

.PHONY : CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.o.requires

CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.o.provides: CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.o.requires
	$(MAKE) -f CMakeFiles/tarea2_parte1.dir/build.make CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.o.provides.build
.PHONY : CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.o.provides

CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.o.provides.build: CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.o


CMakeFiles/tarea2_parte1.dir/sift.cpp.o: CMakeFiles/tarea2_parte1.dir/flags.make
CMakeFiles/tarea2_parte1.dir/sift.cpp.o: ../sift.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luis/Escritorio/tarea2/parte1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/tarea2_parte1.dir/sift.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tarea2_parte1.dir/sift.cpp.o -c /home/luis/Escritorio/tarea2/parte1/sift.cpp

CMakeFiles/tarea2_parte1.dir/sift.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tarea2_parte1.dir/sift.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/luis/Escritorio/tarea2/parte1/sift.cpp > CMakeFiles/tarea2_parte1.dir/sift.cpp.i

CMakeFiles/tarea2_parte1.dir/sift.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tarea2_parte1.dir/sift.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/luis/Escritorio/tarea2/parte1/sift.cpp -o CMakeFiles/tarea2_parte1.dir/sift.cpp.s

CMakeFiles/tarea2_parte1.dir/sift.cpp.o.requires:

.PHONY : CMakeFiles/tarea2_parte1.dir/sift.cpp.o.requires

CMakeFiles/tarea2_parte1.dir/sift.cpp.o.provides: CMakeFiles/tarea2_parte1.dir/sift.cpp.o.requires
	$(MAKE) -f CMakeFiles/tarea2_parte1.dir/build.make CMakeFiles/tarea2_parte1.dir/sift.cpp.o.provides.build
.PHONY : CMakeFiles/tarea2_parte1.dir/sift.cpp.o.provides

CMakeFiles/tarea2_parte1.dir/sift.cpp.o.provides.build: CMakeFiles/tarea2_parte1.dir/sift.cpp.o


# Object files for target tarea2_parte1
tarea2_parte1_OBJECTS = \
"CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.o" \
"CMakeFiles/tarea2_parte1.dir/sift.cpp.o"

# External object files for target tarea2_parte1
tarea2_parte1_EXTERNAL_OBJECTS =

tarea2_parte1: CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.o
tarea2_parte1: CMakeFiles/tarea2_parte1.dir/sift.cpp.o
tarea2_parte1: CMakeFiles/tarea2_parte1.dir/build.make
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_xphoto.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_xobjdetect.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_tracking.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_surface_matching.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_structured_light.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_stereo.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_saliency.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_rgbd.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_reg.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_plot.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_optflow.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_line_descriptor.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_fuzzy.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_dpm.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_dnn.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_datasets.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_ccalib.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_bioinspired.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_bgsegm.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_aruco.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_videostab.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_superres.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_stitching.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_photo.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_text.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_face.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_ximgproc.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_xfeatures2d.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_shape.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_video.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_objdetect.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_calib3d.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_features2d.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_ml.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_highgui.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_videoio.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_imgcodecs.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_imgproc.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_flann.so.3.1.0
tarea2_parte1: /home/luis/anaconda3/lib/libopencv_core.so.3.1.0
tarea2_parte1: CMakeFiles/tarea2_parte1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/luis/Escritorio/tarea2/parte1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable tarea2_parte1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tarea2_parte1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tarea2_parte1.dir/build: tarea2_parte1

.PHONY : CMakeFiles/tarea2_parte1.dir/build

CMakeFiles/tarea2_parte1.dir/requires: CMakeFiles/tarea2_parte1.dir/main_parte1.cpp.o.requires
CMakeFiles/tarea2_parte1.dir/requires: CMakeFiles/tarea2_parte1.dir/sift.cpp.o.requires

.PHONY : CMakeFiles/tarea2_parte1.dir/requires

CMakeFiles/tarea2_parte1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tarea2_parte1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tarea2_parte1.dir/clean

CMakeFiles/tarea2_parte1.dir/depend:
	cd /home/luis/Escritorio/tarea2/parte1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/luis/Escritorio/tarea2/parte1 /home/luis/Escritorio/tarea2/parte1 /home/luis/Escritorio/tarea2/parte1/build /home/luis/Escritorio/tarea2/parte1/build /home/luis/Escritorio/tarea2/parte1/build/CMakeFiles/tarea2_parte1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tarea2_parte1.dir/depend
