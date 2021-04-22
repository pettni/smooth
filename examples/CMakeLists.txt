include(FetchContent)

fetchcontent_declare(matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
        GIT_SHALLOW    ON
        GIT_TAG        origin/master
)

FetchContent_GetProperties(matplotplusplus)
if(NOT matplotplusplus_POPULATED)
    FetchContent_Populate(matplotplusplus)
    add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_executable(bspline bspline.cpp)
target_link_libraries(bspline smooth matplot)

add_executable(odeint odeint.cpp)
target_link_libraries(odeint smooth matplot)