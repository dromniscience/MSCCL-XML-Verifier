cmake_minimum_required(VERSION 3.10)
project(xml-verifier LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

set(COMMON_SOURCES
    src/common/threadblock.cpp
    src/common/mailbox.cpp
    src/common/instructions.cpp
    src/common/tinyxml2.cpp
)

add_library(verifier_core STATIC ${COMMON_SOURCES})
target_include_directories(verifier_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(verifier_core PUBLIC Threads::Threads)
target_compile_features(verifier_core PUBLIC cxx_std_17)

function(add_verifier name)
  add_executable(${name} "src/${name}.cpp")
  target_link_libraries(${name} PRIVATE verifier_core)
endfunction()

add_verifier(allgather-verifier)
add_verifier(alltoall-verifier)
add_verifier(alltoallv-verifier)