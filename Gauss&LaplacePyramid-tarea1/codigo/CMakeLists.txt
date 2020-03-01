cmake_minimum_required (VERSION 2.6)
project (TAREA1EL7008) 
include_directories (include) 
find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIBRARY_DIRS})
add_executable (tarea1 tarea1.cpp)
target_link_libraries( tarea1 ${OpenCV_LIBS} )

