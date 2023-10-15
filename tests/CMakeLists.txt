include(GoogleTest)
find_package(GTest REQUIRED)

option(ENABLE_TEST_COVERAGE "Enable code coverage for debug build" OFF)
message(STATUS "ENABLE_TEST_COVERAGE set to: ${ENABLE_TEST_COVERAGE}")

option(ENABLE_TEST_SANITIZERS "Enable santitzers for debug build" OFF)
message(STATUS "ENABLE_TEST_SANITIZERS set to: ${ENABLE_TEST_SANITIZERS}")

find_package(fmt QUIET)

configure_file(
  ${CMAKE_ROOT}/Modules/DartConfiguration.tcl.in ${PROJECT_BINARY_DIR}/DartConfiguration.tcl
)

add_library(TestConfig INTERFACE)
target_compile_options(
  TestConfig
  INTERFACE -Wall
            -Wcast-align
            -Wconversion
            -Wdouble-promotion
            -Werror
            -Wextra
            -Wimplicit-fallthrough
            -Wnon-virtual-dtor
            -Wnull-dereference
            -Wold-style-cast
            -Woverloaded-virtual
            -Wpedantic
            -Wshadow
            -Wsign-conversion
            -Wunused
            -fdiagnostics-color=always
)
target_link_libraries(TestConfig INTERFACE smooth GTest::Main)
if(fmt_FOUND)
  message(STATUS "fmt found")
  target_link_libraries(TestConfig INTERFACE fmt::fmt)
  target_compile_definitions(TestConfig INTERFACE SMOOTH_HAS_FMT)
else()
  message(STATUS "fmt NOT found")
endif()

if(ENABLE_TEST_COVERAGE)
  target_compile_options(TestConfig INTERFACE $<$<CONFIG:Debug>:--coverage>)
  target_link_libraries(TestConfig INTERFACE $<$<CONFIG:Debug>:--coverage>)
endif()

if(ENABLE_TEST_SANITIZERS)
  set(SANITIZERS "address,leak,undefined")
  target_compile_options(TestConfig INTERFACE $<$<CONFIG:Debug>:-fsanitize=${SANITIZERS}>)
  target_link_options(TestConfig INTERFACE $<$<CONFIG:Debug>:-fsanitize=${SANITIZERS}>)
endif()

function(add_smooth_test name)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs LINK_LIBRARIES)
  cmake_parse_arguments(
    add_smooth_test
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )
  message(STATUS "adding test '${name}' (extra libraries '${add_smooth_test_LINK_LIBRARIES}')")
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE TestConfig ${add_smooth_test_LINK_LIBRARIES})
  gtest_discover_tests(${name})
endfunction()

if(ENABLE_TEST_COVERAGE)
  target_compile_options(TestConfig INTERFACE $<$<CONFIG:Debug>:--coverage>)
  target_link_libraries(TestConfig INTERFACE $<$<CONFIG:Debug>:--coverage>)
endif()

if(ENABLE_TEST_SANITIZERS)
  target_compile_options(TestConfig INTERFACE $<$<CONFIG:Debug>:-fsanitize=undefined,address>)
  target_link_options(TestConfig INTERFACE $<$<CONFIG:Debug>:-fsanitize=undefined,address>)
endif()

# CORE TESTS

add_smooth_test(test_adapted)
add_smooth_test(test_bundle)
add_smooth_test(test_c1)
add_smooth_test(test_galilei)
add_smooth_test(test_lie_api)
add_smooth_test(test_lie_dynamic)
add_smooth_test(test_manifold_any)
add_smooth_test(test_manifold_sub)
add_smooth_test(test_manifold_variant)
add_smooth_test(test_manifold_vector)
add_smooth_test(test_polynomial)
add_smooth_test(test_se2)
add_smooth_test(test_se3)
add_smooth_test(test_se_k_3)
add_smooth_test(test_so2)
add_smooth_test(test_so3)
add_smooth_test(test_utils)

# ALGO TESTS

add_smooth_test(test_cspline)
add_smooth_test(test_diff)
add_smooth_test(test_diff_analytic)
add_smooth_test(test_hessian)
add_smooth_test(test_jacobians)
add_smooth_test(test_nlreg)
add_smooth_test(test_nls)
add_smooth_test(test_optim)
add_smooth_test(test_sparse)
add_smooth_test(test_spline)
add_smooth_test(test_spline_dubins)
add_smooth_test(test_spline_fit)
add_smooth_test(test_spline_reparameterize)

# COMPATABILITY TESTS

find_package(autodiff QUIET)
if(autodiff_FOUND)
  target_compile_definitions(test_diff PRIVATE ENABLE_AUTODIFF_TESTS)
  target_compile_definitions(test_spline PRIVATE ENABLE_AUTODIFF_TESTS)
  target_compile_definitions(test_cspline PRIVATE ENABLE_AUTODIFF_TESTS)
  target_compile_definitions(test_hessian PRIVATE ENABLE_AUTODIFF_TESTS)

  target_link_libraries(test_diff PRIVATE autodiff::autodiff)
  target_link_libraries(test_spline PRIVATE autodiff::autodiff)
  target_link_libraries(test_cspline PRIVATE autodiff::autodiff)
  target_link_libraries(test_hessian PRIVATE autodiff::autodiff)

  add_smooth_test(test_d2rexp LINK_LIBRARIES autodiff::autodiff)
else()
  message(WARNING "autodiff not found, disabling autodiff tests")
endif()

find_package(Boost QUIET)
if(Boost_FOUND)
  add_smooth_test(test_odeint LINK_LIBRARIES Boost::headers)
else()
  message(WARNING "Boost not found, disabling odeint test")
endif()

find_package(Ceres 2.1 QUIET)
if(Ceres_FOUND)
  target_compile_definitions(test_diff PRIVATE ENABLE_CERESDIFF_TESTS)
  target_link_libraries(test_diff PRIVATE Ceres::ceres)

  add_smooth_test(test_ceres LINK_LIBRARIES Ceres::ceres)
else()
  message(WARNING "ceres not found, disabling ceres tests")
endif()

find_package(geometry_msgs QUIET)
if(geometry_msgs_FOUND)
  add_smooth_test(test_ros LINK_LIBRARIES geometry_msgs::geometry_msgs__rosidl_typesupport_cpp)
else()
  message(WARNING "geometry_msgs not found, disabling ros test")
endif()
