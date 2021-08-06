add_executable(bspline bspline.cpp)
target_link_libraries(bspline smooth)

add_executable(spline_fit spline_fit.cpp)
target_link_libraries(spline_fit smooth)

add_executable(odeint odeint.cpp)
target_link_libraries(odeint smooth)

add_executable(odeint_diff odeint_diff.cpp)
target_link_libraries(odeint_diff smooth)

add_executable(nls nls.cpp)
target_link_libraries(nls smooth)

add_executable(snippets snippets.cpp)
target_link_libraries(snippets smooth)

find_package(Matplot++ QUIET)
if(${Matplot++_FOUND})
  set(CMAKE_CXX_STANDARD 20)  # thanks Matplot++
  target_compile_definitions(bspline PRIVATE ENABLE_PLOTTING)
  target_link_libraries(bspline Matplot++::matplot)

  target_compile_definitions(spline_fit PRIVATE ENABLE_PLOTTING)
  target_link_libraries(spline_fit Matplot++::matplot)

  target_compile_definitions(odeint PRIVATE ENABLE_PLOTTING)
  target_link_libraries(odeint Matplot++::matplot)

  target_compile_definitions(odeint_diff PRIVATE ENABLE_PLOTTING)
  target_link_libraries(odeint_diff Matplot++::matplot)
else()
  message(WARNING "Matplot++ not found, plotting disabled")
endif()
