include(FetchContent)

fetchcontent_declare(matplotplusplus
  GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
  GIT_SHALLOW    ON
  GIT_TAG        origin/master
)

fetchcontent_getproperties(matplotplusplus)
if(NOT ${matplotplusplus_POPULATED})
  fetchcontent_populate(matplotplusplus)
  add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_executable(bspline bspline.cpp)
target_link_libraries(bspline smooth matplot)

add_executable(bspline_fit bspline_fit.cpp)
target_link_libraries(bspline_fit smooth matplot)

add_executable(odeint odeint.cpp)
target_link_libraries(odeint smooth matplot)

add_executable(nls nls.cpp)
target_link_libraries(nls smooth)

include(cmake/FindCHOLMOD.cmake)
include(cmake/FindSPQR.cmake)

# add_executable(sparse sparse.cpp)
# target_include_directories(sparse PRIVATE /usr/include/suitesparse)
# target_link_libraries(sparse smooth ${SPQR_LIBRARIES})
