cmake_minimum_required(VERSION 3.0)

project(example)

file(GLOB SOURCES inference_time.cpp model_loader.cpp)

add_executable(inference_time ${SOURCES})


find_package(TensorflowCC REQUIRED)
target_link_libraries(inference_time TensorflowCC::TensorflowCC)

find_package(OpenCV REQUIRED)
target_link_libraries(inference_time ${OpenCV_LIBRARIES})

find_package(Protobuf REQUIRED)
target_link_libraries(inference_time ${Protobuf_LIBRARIES})

find_package(CUDA)
if(CUDA_FOUND)
  target_link_libraries(inference_time ${CUDA_LIBRARIES})
endif()
