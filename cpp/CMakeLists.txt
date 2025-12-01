cmake_minimum_required(VERSION 3.18)
project(alphazero_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ENV{TORCH_DIR})
  message(FATAL_ERROR "Please set TORCH_DIR env var to your LibTorch directory (e.g., export TORCH_DIR=/path/to/libtorch)")
endif()

set(TORCH_DIR $ENV{TORCH_DIR})
set(CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)

add_executable(alphazero_cpp
  src/main.cpp
  src/chess.cpp
  src/mcts.cpp
  src/inference.cpp
)

target_include_directories(alphazero_cpp PRIVATE ${TORCH_INCLUDE_DIRS} src)
target_link_libraries(alphazero_cpp "${TORCH_LIBRARIES}")

set_property(TARGET alphazero_cpp PROPERTY CXX_STANDARD 17)
