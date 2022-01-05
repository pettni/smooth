include(GoogleTest)
find_package(GTest REQUIRED)

configure_file(
  ${CMAKE_ROOT}/Modules/DartConfiguration.tcl.in
  ${PROJECT_BINARY_DIR}/DartConfiguration.tcl
)

add_compile_options(-Wall -Wextra -Wpedantic -Werror)

# CORE TESTS

add_executable(test_utils test_utils.cpp)
target_link_libraries(test_utils PRIVATE smooth GTest::Main)
gtest_discover_tests(test_utils)

add_executable(test_polynomial test_polynomial.cpp)
target_link_libraries(test_polynomial PRIVATE smooth GTest::Main)
gtest_discover_tests(test_polynomial)

add_executable(test_lie_api test_liegroup.cpp)
target_link_libraries(test_lie_api PRIVATE smooth GTest::Main)
gtest_discover_tests(test_lie_api)

add_executable(test_lie_dynamic test_liegroup_dynamic.cpp)
target_link_libraries(test_lie_dynamic PRIVATE smooth GTest::Main)
gtest_discover_tests(test_lie_dynamic)

add_executable(test_so2 test_so2.cpp)
target_link_libraries(test_so2 PRIVATE smooth GTest::Main)
gtest_discover_tests(test_so2)

add_executable(test_so3 test_so3.cpp)
target_link_libraries(test_so3 PRIVATE smooth GTest::Main)
gtest_discover_tests(test_so3)

add_executable(test_se2 test_se2.cpp)
target_link_libraries(test_se2 PRIVATE smooth GTest::Main)
gtest_discover_tests(test_se2)

add_executable(test_se3 test_se3.cpp)
target_link_libraries(test_se3 PRIVATE smooth GTest::Main)
gtest_discover_tests(test_se3)

add_executable(test_c1 test_c1.cpp)
target_link_libraries(test_c1 PRIVATE smooth GTest::Main)
gtest_discover_tests(test_c1)

add_executable(test_bundle test_bundle.cpp)
target_link_libraries(test_bundle PRIVATE smooth GTest::Main)
gtest_discover_tests(test_bundle)

add_executable(test_adapted test_adapted.cpp)
target_link_libraries(test_adapted PRIVATE smooth GTest::Main)
gtest_discover_tests(test_adapted)

# ALGO TESTS

add_executable(test_cspline test_cspline.cpp)
target_link_libraries(test_cspline PRIVATE smooth GTest::Main)
gtest_discover_tests(test_cspline)

add_executable(test_spline test_spline.cpp)
target_link_libraries(test_spline PRIVATE smooth GTest::Main)
gtest_discover_tests(test_spline)

add_executable(test_spline_fit test_spline_fit.cpp)
target_link_libraries(test_spline_fit PRIVATE smooth GTest::Main)
gtest_discover_tests(test_spline_fit)

add_executable(test_spline_dubins test_spline_dubins.cpp)
target_link_libraries(test_spline_dubins PRIVATE smooth GTest::Main)
gtest_discover_tests(test_spline_dubins)

add_executable(test_spline_reparameterize test_spline_reparameterize.cpp)
target_link_libraries(test_spline_reparameterize PRIVATE smooth GTest::Main)
gtest_discover_tests(test_spline_reparameterize)

add_executable(test_diff test_diff.cpp)
target_link_libraries(test_diff PRIVATE smooth GTest::Main)
gtest_discover_tests(test_diff)

add_executable(test_nls test_nls.cpp)
target_link_libraries(test_nls PRIVATE smooth GTest::Main)
gtest_discover_tests(test_nls)

add_executable(test_nlreg test_nlreg.cpp)
target_link_libraries(test_nlreg PRIVATE smooth GTest::Main)
gtest_discover_tests(test_nlreg)

# COMPATABILITY TESTS

find_package(Boost QUIET)
find_package(Ceres 2 QUIET)
find_package(autodiff 0.6 QUIET)
find_package(geometry_msgs QUIET)

if(autodiff_FOUND)
  add_executable(test_hessian test_hessian.cpp)
  target_link_libraries(test_hessian PRIVATE smooth GTest::Main autodiff::autodiff)
  gtest_discover_tests(test_hessian)

  target_compile_definitions(test_diff PRIVATE ENABLE_AUTODIFF_TESTS)
  target_compile_definitions(test_spline PRIVATE ENABLE_AUTODIFF_TESTS)
  target_compile_definitions(test_cspline PRIVATE ENABLE_AUTODIFF_TESTS)

  target_link_libraries(test_diff PRIVATE autodiff::autodiff)
  target_link_libraries(test_spline PRIVATE autodiff::autodiff)
  target_link_libraries(test_cspline PRIVATE autodiff::autodiff)
else()
  message(WARNING "autodiff not found, disabling autodiff tests")
endif()

if(Boost_FOUND)
  add_executable(test_odeint test_odeint.cpp)
  target_link_libraries(test_odeint PRIVATE smooth Boost::headers GTest::Main)
  gtest_discover_tests(test_odeint)
else()
  message(WARNING "Boost not found, disabling odeint test")
endif()

if(Ceres_FOUND)
  target_compile_definitions(test_diff PRIVATE ENABLE_CERESDIFF_TESTS)

  target_link_libraries(test_diff PRIVATE Ceres::ceres)

  add_executable(test_ceres test_ceres.cpp)
  target_link_libraries(test_ceres PRIVATE smooth Ceres::ceres GTest::Main)
  gtest_discover_tests(test_ceres)
else()
  message(WARNING "ceres not found, disabling ceres tests")
endif()

if(geometry_msgs_FOUND)
  add_executable(test_ros test_ros.cpp)
  target_link_libraries(
    test_ros
    PRIVATE smooth geometry_msgs::geometry_msgs__rosidl_typesupport_cpp
            GTest::Main)
  gtest_discover_tests(test_ros)
else()
  message(WARNING "geometry_msgs not found, disabling ros test")
endif()
