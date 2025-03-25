
cmake_minimum_required(VERSION 3.12)

project(pipeline C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)


add_executable(pipeline_test "")

target_include_directories(pipeline_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  pipeline_test
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(pipeline_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/pipeline_int8_model_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/pipeline_test.cpp
)

target_link_libraries(
  pipeline_test
  pico-tflmicro
  hardware_pwm
  pico-tflmicro_test
)

pico_enable_stdio_usb(pipeline_test 1)
pico_enable_stdio_uart(pipeline_test 0)

pico_add_extra_outputs(pipeline_test)


add_executable(pipeline "")

target_include_directories(pipeline
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  pipeline
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(pipeline
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/constants.cpp
  ${CMAKE_CURRENT_LIST_DIR}/pipeline_int8_model_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/constants.h
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.h
)

target_link_libraries(
  pipeline
  pico-tflmicro
  hardware_pwm
)

pico_enable_stdio_usb(pipeline 1)
pico_enable_stdio_uart(pipeline 0)

pico_add_extra_outputs(pipeline)

