cmake_minimum_required (VERSION 2.6)
project (TAREA2EL7008_PARTE1) 
find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIBRARY_DIRS})
add_executable (tarea2_parte1 main_parte1.cpp sift.cpp)
target_link_libraries( tarea2_parte1 ${OpenCV_LIBS} )

