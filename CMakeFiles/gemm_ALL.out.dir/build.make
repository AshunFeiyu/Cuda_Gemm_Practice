# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data/data3/zhangys/MyProject_CUDA/My_CUDA_gemm_All_version

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/data3/zhangys/MyProject_CUDA/My_CUDA_gemm_All_version

# Include any dependencies generated for this target.
include CMakeFiles/gemm_ALL.out.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/gemm_ALL.out.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/gemm_ALL.out.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gemm_ALL.out.dir/flags.make

CMakeFiles/gemm_ALL.out.dir/gemm_ALL.cu.o: CMakeFiles/gemm_ALL.out.dir/flags.make
CMakeFiles/gemm_ALL.out.dir/gemm_ALL.cu.o: gemm_ALL.cu
CMakeFiles/gemm_ALL.out.dir/gemm_ALL.cu.o: CMakeFiles/gemm_ALL.out.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data/data3/zhangys/MyProject_CUDA/My_CUDA_gemm_All_version/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/gemm_ALL.out.dir/gemm_ALL.cu.o"
	/usr/local/cuda-11.6/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/gemm_ALL.out.dir/gemm_ALL.cu.o -MF CMakeFiles/gemm_ALL.out.dir/gemm_ALL.cu.o.d -x cu -rdc=true -c /data/data3/zhangys/MyProject_CUDA/My_CUDA_gemm_All_version/gemm_ALL.cu -o CMakeFiles/gemm_ALL.out.dir/gemm_ALL.cu.o

CMakeFiles/gemm_ALL.out.dir/gemm_ALL.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/gemm_ALL.out.dir/gemm_ALL.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/gemm_ALL.out.dir/gemm_ALL.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/gemm_ALL.out.dir/gemm_ALL.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target gemm_ALL.out
gemm_ALL_out_OBJECTS = \
"CMakeFiles/gemm_ALL.out.dir/gemm_ALL.cu.o"

# External object files for target gemm_ALL.out
gemm_ALL_out_EXTERNAL_OBJECTS =

CMakeFiles/gemm_ALL.out.dir/cmake_device_link.o: CMakeFiles/gemm_ALL.out.dir/gemm_ALL.cu.o
CMakeFiles/gemm_ALL.out.dir/cmake_device_link.o: CMakeFiles/gemm_ALL.out.dir/build.make
CMakeFiles/gemm_ALL.out.dir/cmake_device_link.o: /usr/local/cuda-11.6/lib64/libcublas.so
CMakeFiles/gemm_ALL.out.dir/cmake_device_link.o: CMakeFiles/gemm_ALL.out.dir/deviceLinkLibs.rsp
CMakeFiles/gemm_ALL.out.dir/cmake_device_link.o: CMakeFiles/gemm_ALL.out.dir/deviceObjects1.rsp
CMakeFiles/gemm_ALL.out.dir/cmake_device_link.o: CMakeFiles/gemm_ALL.out.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/data/data3/zhangys/MyProject_CUDA/My_CUDA_gemm_All_version/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/gemm_ALL.out.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gemm_ALL.out.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gemm_ALL.out.dir/build: CMakeFiles/gemm_ALL.out.dir/cmake_device_link.o
.PHONY : CMakeFiles/gemm_ALL.out.dir/build

# Object files for target gemm_ALL.out
gemm_ALL_out_OBJECTS = \
"CMakeFiles/gemm_ALL.out.dir/gemm_ALL.cu.o"

# External object files for target gemm_ALL.out
gemm_ALL_out_EXTERNAL_OBJECTS =

gemm_ALL.out: CMakeFiles/gemm_ALL.out.dir/gemm_ALL.cu.o
gemm_ALL.out: CMakeFiles/gemm_ALL.out.dir/build.make
gemm_ALL.out: /usr/local/cuda-11.6/lib64/libcublas.so
gemm_ALL.out: CMakeFiles/gemm_ALL.out.dir/cmake_device_link.o
gemm_ALL.out: CMakeFiles/gemm_ALL.out.dir/linkLibs.rsp
gemm_ALL.out: CMakeFiles/gemm_ALL.out.dir/objects1.rsp
gemm_ALL.out: CMakeFiles/gemm_ALL.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/data/data3/zhangys/MyProject_CUDA/My_CUDA_gemm_All_version/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable gemm_ALL.out"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gemm_ALL.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gemm_ALL.out.dir/build: gemm_ALL.out
.PHONY : CMakeFiles/gemm_ALL.out.dir/build

CMakeFiles/gemm_ALL.out.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gemm_ALL.out.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gemm_ALL.out.dir/clean

CMakeFiles/gemm_ALL.out.dir/depend:
	cd /data/data3/zhangys/MyProject_CUDA/My_CUDA_gemm_All_version && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/data3/zhangys/MyProject_CUDA/My_CUDA_gemm_All_version /data/data3/zhangys/MyProject_CUDA/My_CUDA_gemm_All_version /data/data3/zhangys/MyProject_CUDA/My_CUDA_gemm_All_version /data/data3/zhangys/MyProject_CUDA/My_CUDA_gemm_All_version /data/data3/zhangys/MyProject_CUDA/My_CUDA_gemm_All_version/CMakeFiles/gemm_ALL.out.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/gemm_ALL.out.dir/depend

