cmake_minimum_required(VERSION 2.8.3)
project(mrsmap_ros)

################################################
## CMake Settings for Original MRSMap Library ##
################################################

# In case the user does not setup CMAKE_BUILD_TYPE, assume it's RelWithDebInfo
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "build type default to RelWithDebInfo, set to Release to improve performance" FORCE)
endif("${CMAKE_BUILD_TYPE}" STREQUAL "")

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  message(STATUS "building in release mode")
  add_definitions( -msse4.2 -fopenmp )
  add_definitions("-DBOOST_DISABLE_ASSERTS -DEIGEN_NO_DEBUG -DNDEBUG")
  add_definitions (-Wall -fno-strict-aliasing)# -fvisibility=hidden -fvisibility-inlines-hidden)
  add_definitions(-Wall -funroll-loops -momit-leaf-frame-pointer -fomit-frame-pointer -floop-block -ftree-loop-distribution -ftree-loop-linear -floop-interchange -floop-strip-mine -fgcse-lm -fgcse-sm -fsched-spec-load)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/mrsmap_ros/cmake/Modules/" ${CMAKE_MODULE_PATH})

# catkin_make system
find_package(catkin REQUIRED COMPONENTS roscpp sensor_msgs tf)

# PCL required
set(OPENNI_INCLUDE_DIRS /usr/include/ni)
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# OctreeLib required
find_package(OctreeLib REQUIRED)
include_directories(${OCTREELIB_INCLUDE_DIRS})

# GSL required
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})
link_directories(${GSL_LIBRARY_DIRS})

# TBB required
find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})

# OpenCV required
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# G2O required
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIR})

find_package( Boost 1.40 COMPONENTS program_options REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )

# For building the CHOLMOD / CSPARSE solvers
find_package(Cholmod)
find_package(BLAS)
find_package(LAPACK)
find_package(CSparse)
include_directories(${CSPARSE_INCLUDE_DIR})

include_directories(${PROJECT_SOURCE_DIR}/include)

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs  # Or other packages containing msgs
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES mrsmap_ros
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)

## Declare a cpp library
# add_library(mrsmap_ros
#   src/${PROJECT_NAME}/mrsmap_ros.cpp
# )

# -- mrslib for map aggregation, registration, and visualization
set( SRC_MRSMAPLIB    
                src/map/multiresolution_csurfel_map.cpp
                src/registration/multiresolution_csurfel_registration.cpp
		src/utilities/utilities.cpp
		src/visualization/visualization_map.cpp
)

add_library( mrsmap SHARED ${SRC_MRSMAPLIB} )
add_definitions( ${SSE_FLAGS} )
target_link_libraries( mrsmap octreelib ${PCL_LIBRARIES} ${OpenCV_LIBS} ${TBB_LIBRARIES} ${GSL_LIBRARIES})


# -- mrsslamlib for SLAM using mrs maps
add_library( mrsslam SHARED src/slam/slam.cpp src/visualization/visualization_slam.cpp )
add_definitions( ${SSE_FLAGS} )
target_link_libraries( mrsslam mrsmap ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_CSPARSE_EXTENSION} ${G2O_TYPES_SLAM3D})


## Declare a cpp executable
# add_executable(mrsmap_ros_node src/mrsmap_ros_node.cpp)

add_executable( mrsslam_app src/apps/mrsslam_app.cpp )
add_definitions( ${SSE_FLAGS} )
target_link_libraries( mrsslam_app mrsslam ${catkin_LIBRARIES})

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(mrsmap_ros_node mrsmap_ros_generate_messages_cpp)

## Specify libraries to link a library or executable target against
# target_link_libraries(mrsmap_ros_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS mrsmap_ros mrsmap_ros_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_mrsmap_ros.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
