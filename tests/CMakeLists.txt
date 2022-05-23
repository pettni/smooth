include(GoogleTest)
find_package(GTest REQUIRED)

configure_file(
  ${CMAKE_ROOT}/Modules/DartConfiguration.tcl.in
  ${PROJECT_BINARY_DIR}/DartConfiguration.tcl
)

add_library(TestConfig INTERFACE)
target_compile_options(
  TestConfig
  INTERFACE
  -Wall
  -Wextra
  -Wshadow
  -Wnon-virtual-dtor
  -Wold-style-cast
  -Wcast-align
  -Wunused
  -Woverloaded-virtual
  -Wpedantic
  -Wconversion
  -Wsign-conversion
  -Wnull-dereference
  -Wdouble-promotion
  -Wimplicit-fallthrough
  # -Werror
)
target_link_libraries(
  TestConfig
  INTERFACE
  smooth
  GTest::Main
  -fsanitize=undefined,address
)

# CORE TESTS

add_executable(test_utils test_utils.cpp)
target_link_libraries(test_utils PRIVATE TestConfig)
gtest_discover_tests(test_utils)

add_executable(test_polynomial test_polynomial.cpp)
target_link_libraries(test_polynomial PRIVATE TestConfig)
gtest_discover_tests(test_polynomial)

add_executable(test_lie_api test_liegroup.cpp)
target_link_libraries(test_lie_api PRIVATE TestConfig)
gtest_discover_tests(test_lie_api)

add_executable(test_lie_dynamic test_liegroup_dynamic.cpp)
target_link_libraries(test_lie_dynamic PRIVATE TestConfig)
gtest_discover_tests(test_lie_dynamic)

add_executable(test_so2 test_so2.cpp)
target_link_libraries(test_so2 PRIVATE TestConfig)
gtest_discover_tests(test_so2)

add_executable(test_so3 test_so3.cpp)
target_link_libraries(test_so3 PRIVATE TestConfig)
gtest_discover_tests(test_so3)

add_executable(test_se2 test_se2.cpp)
target_link_libraries(test_se2 PRIVATE TestConfig)
gtest_discover_tests(test_se2)

add_executable(test_se3 test_se3.cpp)
target_link_libraries(test_se3 PRIVATE TestConfig)
gtest_discover_tests(test_se3)

add_executable(test_galilei test_galilei.cpp)
target_link_libraries(test_galilei PRIVATE TestConfig)
gtest_discover_tests(test_galilei)

add_executable(test_c1 test_c1.cpp)
target_link_libraries(test_c1 PRIVATE TestConfig)
gtest_discover_tests(test_c1)

add_executable(test_bundle test_bundle.cpp)
target_link_libraries(test_bundle PRIVATE TestConfig)
gtest_discover_tests(test_bundle)

add_executable(test_adapted test_adapted.cpp)
target_link_libraries(test_adapted PRIVATE TestConfig)
gtest_discover_tests(test_adapted)

add_executable(test_manifold_vector test_manifold_vector.cpp)
target_link_libraries(test_manifold_vector PRIVATE TestConfig)
gtest_discover_tests(test_manifold_vector)

add_executable(test_manifold_variant test_manifold_variant.cpp)
target_link_libraries(test_manifold_variant PRIVATE TestConfig)
gtest_discover_tests(test_manifold_variant)

# ALGO TESTS

add_executable(test_cspline test_cspline.cpp)
target_link_libraries(test_cspline PRIVATE TestConfig)
gtest_discover_tests(test_cspline)

add_executable(test_spline test_spline.cpp)
target_link_libraries(test_spline PRIVATE TestConfig)
gtest_discover_tests(test_spline)

add_executable(test_spline_fit test_spline_fit.cpp)
target_link_libraries(test_spline_fit PRIVATE TestConfig)
gtest_discover_tests(test_spline_fit)

add_executable(test_spline_dubins test_spline_dubins.cpp)
target_link_libraries(test_spline_dubins PRIVATE TestConfig)
gtest_discover_tests(test_spline_dubins)

add_executable(test_spline_reparameterize test_spline_reparameterize.cpp)
target_link_libraries(test_spline_reparameterize PRIVATE TestConfig)
gtest_discover_tests(test_spline_reparameterize)

add_executable(test_diff test_diff.cpp)
target_link_libraries(test_diff PRIVATE TestConfig)
gtest_discover_tests(test_diff)

add_executable(test_diff_analytic test_diff_analytic.cpp)
target_link_libraries(test_diff_analytic PRIVATE TestConfig)
gtest_discover_tests(test_diff_analytic)

add_executable(test_nls test_nls.cpp)
target_link_libraries(test_nls PRIVATE TestConfig)
gtest_discover_tests(test_nls)

add_executable(test_nlreg test_nlreg.cpp)
target_link_libraries(test_nlreg PRIVATE TestConfig)
gtest_discover_tests(test_nlreg)

add_executable(test_jacobians test_jacobians.cpp)
target_link_libraries(test_jacobians PRIVATE TestConfig)
gtest_discover_tests(test_jacobians)

add_executable(test_hessian test_hessian.cpp)
target_link_libraries(test_hessian PRIVATE TestConfig)
gtest_discover_tests(test_hessian)

add_executable(test_sparse test_sparse.cpp)
target_link_libraries(test_sparse PRIVATE TestConfig)
gtest_discover_tests(test_sparse)

# COMPATABILITY TESTS

find_package(Boost QUIET)
find_package(Ceres 2.1 QUIET)
find_package(autodiff 0.6 QUIET)
find_package(geometry_msgs QUIET)

if(autodiff_FOUND)
  target_compile_definitions(test_diff PRIVATE ENABLE_AUTODIFF_TESTS)
  target_compile_definitions(test_spline PRIVATE ENABLE_AUTODIFF_TESTS)
  target_compile_definitions(test_cspline PRIVATE ENABLE_AUTODIFF_TESTS)
  target_compile_definitions(test_hessian PRIVATE ENABLE_AUTODIFF_TESTS)

  target_link_libraries(test_diff PRIVATE autodiff::autodiff)
  target_link_libraries(test_spline PRIVATE autodiff::autodiff)
  target_link_libraries(test_cspline PRIVATE autodiff::autodiff)
  target_link_libraries(test_hessian PRIVATE autodiff::autodiff)

  add_executable(test_d2rexp test_d2rexp.cpp)
  target_link_libraries(test_d2rexp PRIVATE TestConfig autodiff::autodiff)
  gtest_discover_tests(test_d2rexp)

else()
  message(WARNING "autodiff not found, disabling autodiff tests")
endif()

if(Boost_FOUND)
  add_executable(test_odeint test_odeint.cpp)
  target_link_libraries(test_odeint PRIVATE TestConfig Boost::headers)
  gtest_discover_tests(test_odeint)
else()
  message(WARNING "Boost not found, disabling odeint test")
endif()

if(Ceres_FOUND)
  target_compile_definitions(test_diff PRIVATE ENABLE_CERESDIFF_TESTS)
  target_link_libraries(test_diff PRIVATE Ceres::ceres)

  add_executable(test_ceres test_ceres.cpp)
  target_link_libraries(test_ceres PRIVATE TestConfig Ceres::ceres)
  gtest_discover_tests(test_ceres)
else()
  message(WARNING "ceres not found, disabling ceres tests")
endif()

if(geometry_msgs_FOUND)
  add_executable(test_ros test_ros.cpp)
  target_link_libraries(test_ros PRIVATE TestConfig geometry_msgs::geometry_msgs__rosidl_typesupport_cpp)
  gtest_discover_tests(test_ros)
else()
  message(WARNING "geometry_msgs not found, disabling ros test")
endif()
