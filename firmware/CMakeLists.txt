file(GLOB RUNTIME_SRCS "${CMAKE_CURRENT_LIST_DIR}/needle_runtime/*.cpp")
file(GLOB APP_SRCS "${CMAKE_CURRENT_LIST_DIR}/storage/*.cpp" "${CMAKE_CURRENT_LIST_DIR}/ui/*.cpp"
                   "${CMAKE_CURRENT_LIST_DIR}/app/*.cpp")
idf_component_register(
    SRCS "main.cpp" ${RUNTIME_SRCS} ${APP_SRCS}
    INCLUDE_DIRS "." "needle_runtime"
    REQUIRES fatfs sdmmc esp_driver_sdspi esp_driver_sdmmc esp_driver_spi esp_driver_gpio
             esp_driver_uart esp_partition spi_flash esp_timer bootloader_support app_update
             vfs esp_system)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-unused-parameter -Wno-missing-field-initializers -DNR_MAX_CHUNK=1)
