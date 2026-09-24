# ESP-IDF project: Needle 3 on the Cheap Yellow Display (ESP32-2432S028R).
#   idf.py set-target esp32 && idf.py build
cmake_minimum_required(VERSION 3.16)
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/firmware")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(needle_cyd)
