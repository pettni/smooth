cmake_minimum_required(VERSION 3.15)
project(smooth VERSION 0.0.1)

include(CTest)
include(GoogleTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_EXAMPLES "Build the examples." OFF)

find_package(Eigen3 REQUIRED)

# ---------------------------------------------------------------------------------------
# TARGETS
# ---------------------------------------------------------------------------------------

add_library(smooth INTERFACE)
target_include_directories(smooth
INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(smooth
INTERFACE
  Eigen3::Eigen
)

# ---------------------------------------------------------------------------------------
# EXAMPLES
# ---------------------------------------------------------------------------------------

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------------------
# INSTALLATION
# ---------------------------------------------------------------------------------------

include(GNUInstallDirs)

# Header files
install(
  DIRECTORY    ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}
  DESTINATION  ${CMAKE_INSTALL_INCLUDEDIR}
)

# Targets
install(
  TARGETS               smooth
  EXPORT                ${PROJECT_NAME}_targets
  LIBRARY DESTINATION   ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION   ${CMAKE_INSTALL_LIBDIR}
)

# ---------------------------------------------------------------------------------------
# CMAKE CONFIGURATION
# ---------------------------------------------------------------------------------------

set(export_dest_dir ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

# Generate TARGETS files
install(
  EXPORT       ${PROJECT_NAME}_targets
  FILE         ${PROJECT_NAME}Targets.cmake
  NAMESPACE    ${PROJECT_NAME}::
  DESTINATION  ${export_dest_dir}
)

# Generate CONFIG file
include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/ConfigTemplate.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${export_dest_dir}
)

# Generate VERSION file
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Version.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY SameMajorVersion
)

# Install CONFIG and VERSION files
install(
  FILES        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Version.cmake
  DESTINATION  ${export_dest_dir}
)

# ---------------------------------------------------------------------------------------
# TESTING
# ---------------------------------------------------------------------------------------

if(BUILD_TESTING)
  find_package(GTest REQUIRED)

  add_compile_options(-Wall -Wextra -Wpedantic -Werror)

  # CORE TESTS

  add_executable(${PROJECT_NAME}_tests
    test/test_liegroup.cpp
    test/test_bundle.cpp
    test/test_so2.cpp
    test/test_so3.cpp
    test/test_se2.cpp
    test/test_se3.cpp
  )
  target_link_libraries(${PROJECT_NAME}_tests PRIVATE smooth GTest::Main)
  gtest_discover_tests(${PROJECT_NAME}_tests)

  add_executable(test_manifold_vector test/test_manifold_vector.cpp)
  target_link_libraries(test_manifold_vector PRIVATE smooth GTest::Main)
  gtest_discover_tests(test_manifold_vector)

  # ALGO TESTS

  add_executable(test_spline test/test_spline.cpp)
  target_link_libraries(test_spline PRIVATE smooth GTest::Main)
  gtest_discover_tests(test_spline)

  add_executable(test_diff test/test_diff.cpp)
  target_link_libraries(test_diff PRIVATE smooth GTest::Main)
  gtest_discover_tests(test_diff)

  add_executable(test_nls test/test_nls.cpp)
  target_link_libraries(test_nls PRIVATE smooth GTest::Main)
  gtest_discover_tests(test_nls)

  add_executable(test_nlreg test/test_nlreg.cpp)
  target_link_libraries(test_nlreg PRIVATE smooth GTest::Main)
  gtest_discover_tests(test_nlreg)

  # COMPATABILITY TESTS

  find_package(Boost QUIET)
  if(Boost_FOUND)
    add_executable(test_odeint test/test_odeint.cpp)
    target_link_libraries(test_odeint PRIVATE smooth Boost::headers GTest::Main)
    gtest_discover_tests(test_odeint)
  else()
    message(WARNING "Boost not found, disabling odeint test")
  endif()

  find_package(Ceres QUIET)
  if(Ceres_FOUND)
    add_executable(test_ceres test/test_ceres.cpp)
    target_link_libraries(test_ceres PRIVATE smooth ${CERES_LIBRARIES} GTest::Main)
    gtest_discover_tests(test_ceres)
  else()
    message(WARNING "ceres not found, disabling odeint test")
  endif()

  find_package(geometry_msgs QUIET)
  if(geometry_msgs_FOUND)
    add_executable(test_ros test/test_ros.cpp)
    target_include_directories(test_ros PRIVATE ${geometry_msgs_INCLUDE_DIRS})
    target_link_libraries(test_ros PRIVATE smooth GTest::Main)
    gtest_discover_tests(test_ros)
  else()
    message(WARNING "geometry_msgs not found, disabling ros test")
  endif()

  find_package(autodiff QUIET)
  if(autodiff_FOUND)
    target_compile_definitions(test_diff PRIVATE ENABLE_AUTODIFF_TESTS)
    target_link_libraries(test_diff PRIVATE autodiff::autodiff)
  else()
    message(WARNING "autodiff not found, disabling autodiff tests")
  endif()
endif()
