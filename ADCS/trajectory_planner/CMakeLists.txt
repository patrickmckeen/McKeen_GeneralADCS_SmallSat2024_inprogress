
cmake_minimum_required(VERSION 3.10)

project(trajectory_planner)


set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)


message(STATUS "using ${CMAKE_CXX_COMPILER}")


#set(CMAKE_BUILD_TYPE Debug)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(BUILD_SHARED_LIBS ON)
set(CMAKE_APPLE_SILICON_PROCESSOR arm64)
#set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE INTERNAL " " FORCE)
set(CMAKE_CXX_FLAGS "-ftemplate-depth=2048 -O3 -fvisibility=hidden")# -fno-math-errno -funsafe-math-optimizations")
set(LDFLAGS "-Wl,-no_fixup_chains")




if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
   message(SEND_ERROR "In-source builds are not allowed.")
endif("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")

#add_subdirectory(pybind11)

FIND_PACKAGE(PythonInterp 3.10 REQUIRED)
FIND_PACKAGE(PythonLibs   3.10 REQUIRED)

#add_subdirectory(armadillo010.2.1)
include_directories(${PYTHON_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIRS})
find_package(pybind11 REQUIRED)
find_package(Catch2 3 REQUIRED)
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

add_library(tp_nopy STATIC
			src/ArmaCSV.cpp
      src/planner/GeneralUtil.cpp
      src/planner/PlannerUtil.cpp
      src/planner/Satellite.cpp
      src/planner/OldPlanner.cpp
			)
set_target_properties(tp_nopy PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(tp STATIC
      src/planner/Satellite.cpp
      src/planner/GeneralUtil.cpp
      src/planner/OldPlanner.cpp
      src/planner/PlannerUtilPy.cpp
      src/planner/PlannerUtil.cpp
      src/ArmaNumpy.cpp
      )
set_target_properties(tp PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(sat STATIC
      src/planner/Satellite.cpp
      src/planner/GeneralUtil.cpp
      src/ArmaNumpy.cpp
      )
set_target_properties(sat PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(sat_nopy STATIC
      src/planner/Satellite.cpp
      src/planner/GeneralUtil.cpp
      )
set_target_properties(sat_nopy PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(tp_test
			   src/planner/main.cpp
			   )
set_target_properties(tp_test PROPERTIES POSITION_INDEPENDENT_CODE ON)


pybind11_add_module(tplaunch MODULE src/planner/PyPlanner.cpp)
pybind11_add_module(pysat MODULE src/planner/Satellite.cpp)


target_link_libraries(tp_nopy PUBLIC ${ARMADILLO_LIBRARIES} ${ARMADILLO_LIBRARY} pybind11::module ${PYBIND11_INCLUDE_DIRS} ${PYTHON_LIBRARY} ${PYTHON_LIBRARIES}  )
target_link_libraries(sat PUBLIC ${ARMADILLO_LIBRARIES} ${ARMADILLO_LIBRARY} pybind11::module ${PYBIND11_INCLUDE_DIRS} ${PYTHON_LIBRARY} ${PYTHON_LIBRARIES}  )
target_link_libraries(sat_nopy PUBLIC ${ARMADILLO_LIBRARIES} ${ARMADILLO_LIBRARY} pybind11::module ${PYBIND11_INCLUDE_DIRS}  ${PYTHON_LIBRARY} ${PYTHON_LIBRARIES} )
target_link_libraries(pysat PUBLIC ${ARMADILLO_LIBRARIES} ${ARMADILLO_LIBRARY} sat ${PYTHON_LIBRARY} ${PYTHON_LIBRARIES}  )
target_link_libraries(tp PUBLIC ${ARMADILLO_LIBRARIES} ${ARMADILLO_LIBRARY} pybind11::module sat ${PYBIND11_INCLUDE_DIRS} ${PYTHON_LIBRARY} ${PYTHON_LIBRARIES}   )
target_link_libraries(tplaunch PUBLIC ${ARMADILLO_LIBRARIES} ${ARMADILLO_LIBRARY} tp tp_nopy sat sat_nopy  pybind11::module ${PYBIND11_INCLUDE_DIRS} ${PYTHON_LIBRARY} ${PYTHON_LIBRARIES}  )
target_link_libraries(tp_test PRIVATE Catch2::Catch2WithMain PUBLIC pybind11::module ${PYBIND11_INCLUDE_DIRS} ${ARMADILLO_LIBRARIES} ${ARMADILLO_LIBRARY} sat tp sat_nopy tp_nopy ${PYTHON_LIBRARY} ${PYTHON_LIBRARIES}   m)


target_include_directories(tp PUBLIC
        "${PROJECT_SOURCE_DIR}/src"
        "${PYTHON_INCLUDE_DIRS}"
        "${PYBIND11_INCLUDE_DIRS}"
        "${ARMADILLO_INCLUDE_DIRS}"
			   )

 target_include_directories(tp_nopy PUBLIC
         "${PROJECT_SOURCE_DIR}/src"
         "${PYTHON_INCLUDE_DIRS}"
         "${PYBIND11_INCLUDE_DIRS}"
         "${ARMADILLO_INCLUDE_DIRS}"
 			   )

 target_include_directories(tplaunch PUBLIC
         "${PROJECT_SOURCE_DIR}/src"
         "${PYTHON_INCLUDE_DIRS}"
         "${PYBIND11_INCLUDE_DIRS}"
         "${ARMADILLO_INCLUDE_DIRS}"
 			   )

 target_include_directories(pysat PRIVATE
         "${PROJECT_SOURCE_DIR}/src"
         "${PYTHON_INCLUDE_DIRS}"
         "${PYBIND11_INCLUDE_DIRS}"
         "${ARMADILLO_INCLUDE_DIRS}"
 			   )

 target_include_directories(sat PUBLIC
         "${PROJECT_SOURCE_DIR}/src"
         "${PYTHON_INCLUDE_DIRS}"
         "${PYBIND11_INCLUDE_DIRS}"
         "${ARMADILLO_INCLUDE_DIRS}"
 			   )


target_include_directories(sat_nopy PUBLIC
        "${PROJECT_SOURCE_DIR}/src"
        "${PYTHON_INCLUDE_DIRS}"
        "${PYBIND11_INCLUDE_DIRS}"
        "${ARMADILLO_INCLUDE_DIRS}"
			   )

target_include_directories(tp_test PUBLIC
			   "${PROJECT_SOURCE_DIR}/src"
 			   ${PROJECT_SOURCE_DIR}
         ${PYTHON_INCLUDE_DIRS}
         ${PYBIND11_INCLUDE_DIRS}
         ${ARMADILLO_INCLUDE_DIRS}
		   )
