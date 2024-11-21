cmake_minimum_required(VERSION 3.16)
project(drake-spawn-body-during-simulation)

find_package(drake CONFIG REQUIRED)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(main main.cc)
target_link_libraries(main drake::drake)
