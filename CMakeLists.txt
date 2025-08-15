cmake_minimum_required(VERSION 3.10)
project(xml-verifier)

set(CMAKE_CXX_STANDARD 17)

add_executable(allgather-verifier src/allgather-verifier.cpp src/threadblock.cpp src/mailbox.cpp src/instructions.cpp src/tinyxml2.cpp)
target_link_libraries(allgather-verifier pthread)