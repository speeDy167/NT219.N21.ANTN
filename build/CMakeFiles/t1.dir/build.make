# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/im5hry/Project_Crypto

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/im5hry/Project_Crypto/build

# Include any dependencies generated for this target.
include CMakeFiles/t1.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/t1.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/t1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/t1.dir/flags.make

CMakeFiles/t1.dir/test.cpp.o: CMakeFiles/t1.dir/flags.make
CMakeFiles/t1.dir/test.cpp.o: ../test.cpp
CMakeFiles/t1.dir/test.cpp.o: CMakeFiles/t1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/im5hry/Project_Crypto/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/t1.dir/test.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/t1.dir/test.cpp.o -MF CMakeFiles/t1.dir/test.cpp.o.d -o CMakeFiles/t1.dir/test.cpp.o -c /home/im5hry/Project_Crypto/test.cpp

CMakeFiles/t1.dir/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/t1.dir/test.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/im5hry/Project_Crypto/test.cpp > CMakeFiles/t1.dir/test.cpp.i

CMakeFiles/t1.dir/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/t1.dir/test.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/im5hry/Project_Crypto/test.cpp -o CMakeFiles/t1.dir/test.cpp.s

# Object files for target t1
t1_OBJECTS = \
"CMakeFiles/t1.dir/test.cpp.o"

# External object files for target t1
t1_EXTERNAL_OBJECTS =

t1: CMakeFiles/t1.dir/test.cpp.o
t1: CMakeFiles/t1.dir/build.make
t1: /usr/local/lib/libOPENFHEpke.so.1.0.3
t1: /usr/local/lib/libOPENFHEbinfhe.so.1.0.3
t1: /usr/local/lib/libOPENFHEcore.so.1.0.3
t1: CMakeFiles/t1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/im5hry/Project_Crypto/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable t1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/t1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/t1.dir/build: t1
.PHONY : CMakeFiles/t1.dir/build

CMakeFiles/t1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/t1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/t1.dir/clean

CMakeFiles/t1.dir/depend:
	cd /home/im5hry/Project_Crypto/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/im5hry/Project_Crypto /home/im5hry/Project_Crypto /home/im5hry/Project_Crypto/build /home/im5hry/Project_Crypto/build /home/im5hry/Project_Crypto/build/CMakeFiles/t1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/t1.dir/depend

