cmake_minimum_required(VERSION 3.0)

project(example)

file(GLOB SOURCES inference.cpp model_loader.cpp)
add_executable(inference ${SOURCES})

find_package(TensorflowCC REQUIRED)
target_link_libraries(inference TensorflowCC::TensorflowCC)

find_package(OpenCV REQUIRED)
target_link_libraries(inference ${OpenCV_LIBRARIES})

find_package(Protobuf REQUIRED)
target_link_libraries(inference ${Protobuf_LIBRARIES})

find_package(CUDA)
if(CUDA_FOUND)
  target_link_libraries(inference ${CUDA_LIBRARIES})
endif()

find_package( OpenCV 3.2.0 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( inference ${OpenCV_LIBS} )