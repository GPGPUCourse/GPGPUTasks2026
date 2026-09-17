cmake_minimum_required(VERSION 3.6)

project(libimages)

# This is thin wrapper around CImg header-only library available at https://github.com/dtschump/CImg (v.2.3.4 0f0d65d984b08ad8178969f4fa4d1641d721354b)

set(HEADERS
        libimages/debug_io.h
        libimages/images.h
        )

set(SOURCES
        libimages/debug_io.cpp
        libimages/images.cpp
        )

set(CMAKE_CXX_STANDARD 17)

set(LIBRARIES libbase)

if(UNIX)
    find_package(X11 REQUIRED)
    find_package(PkgConfig REQUIRED)

    pkg_check_modules(
        GRAPHICSMAGICKPP
        REQUIRED
        IMPORTED_TARGET
        GraphicsMagick++
    )

    list(APPEND LIBRARIES
        ${X11_LIBRARIES}
        PkgConfig::GRAPHICSMAGICKPP
    )
endif()

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})
if(UNIX)
    target_compile_definitions(${PROJECT_NAME} PRIVATE cimg_use_magick)
endif()
