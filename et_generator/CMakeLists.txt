project(et_generator C CXX)

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES $ENV{HOME}"/local/lib")

find_package(Protobuf REQUIRED)
include_directories(
  ${Protobuf_INCLUDE_DIRS}
  ../)

set(TARGET et_generator)
set(SRCS
  et_generator.cpp
  dependency_graph.cpp
  ../et_def/et_def.pb.cc
  ../third_party/utils/protoio.cc)
add_executable(${TARGET} ${SRCS})

target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} ${PROTOBUF_LIBRARIES} -lpthread -lz)
