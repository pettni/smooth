# `smooth`: Lie Theory for Robotics

[![CI Build and Test][ci-shield]][ci-link]
[![Code coverage][cov-shield]][cov-link]
[![License][license-shield]][license-link]

<img src="media/ode.png" width="300">  <img src="media/bspline.png" width="300">

*This project is currently being developed---breaking changes and bugs should be expected.
If you are looking for something stable and established, check out
[manif][manif-link] and [Sophus][sophus-link].*

In robotics it is often convenient to work in non-Euclidean manifolds.
[Lie groups](https://en.wikipedia.org/wiki/Lie_group) are a class of manifolds that are 
easy to work with due to their symmetries, and that are also good models for many robotic
systems. The objective of this header-only C++20 library is to facilitate Lie groups in
robotics software, by enabling:

 * Algebraic manipulation
 * Automatic differentiation
 * Interpolation (right figure shows a B-spline of order 5 on smooth::SO3, see `examples/bspline.cpp`)
 * Numerical integration (left figure shows the solution of an ODE on ![](https://latex.codecogs.com/png.latex?\mathbb{SO}(3)\times\mathbb{R}^3), see `examples/odeint.cpp`)
 * Optimization

The following common Lie groups are implemented:
 * smooth::Tn: n-dimensional translations
 * smooth::SO2: two-dimensional rotations with complex number ![](https://latex.codecogs.com/png.latex?\mathbb{C}(1)) memory representation
 * smooth::SO3: three-dimensional rotations with quaternion ![](https://latex.codecogs.com/png.latex?\mathbb{S}^3) memory representation
 * smooth::SE2: two-dimensional rigid motions
 * smooth::SE3: three-dimensional rigid motions
 * A smooth::Bundle type to treat Lie group products ![](https://latex.codecogs.com/png.latex?G&space;=&space;G_1&space;\times&space;\ldots&space;\times&space;G_n) as a single Lie group. The Bundle type also supports regular Eigen vectors as ![](https://latex.codecogs.com/png.latex?\mathbb{R}^n\cong\mathbb{T}(n)) components

The guiding principles for `smooth` are **brevity, reliability and compatability**. 


## Getting started

### Download and Build

Clone the repository and install it
```bash
git clone https://github.com/pettni/smooth.git
cd smooth
mkdir build && cd build

# Specify a C++20-compatible compiler if your default does not support C++20.
# Build tests and/or examples as desired.
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++-10 -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
make -j8
sudo make install
```

Alternatively, if using ROS or ROS2 just clone `smooth` into a
catkin/[colcon](https://colcon.readthedocs.io/en/released/) workspace source folder and build the
workspace with a compiler that supports C++20. Example with colcon:
```bash
colcon build --cmake-args -DCMAKE_CXX_COMPILER=/usr/bin/g++-10
```

### Use with cmake

To utilize `smooth` in your own project, include something along these lines in your `CMakeLists.txt`
```cmake
find_package(smooth)

add_executable(my_executable main.cpp)
target_link_libraries(my_executable smooth::smooth)
```

### Explore the API

Check out the [Documentation][doc-link] and the [`examples`](https://github.com/pettni/smooth/tree/master/examples).


## Group algebra examples

 ```cpp
 // Also works with other types: SO2d, SE2d, SE3d, Bundle<SO3d, T3d> etc...

 using Tangent = typename smooth::SO3d::Tangent;

 // construct a random group element and a random tangent element
 smooth::SO3d g = smooth::SO3d::Random();
 Tangent a = Tangent::Random();

 // lie group exponential
 auto exp_a = smooth::SO3d::exp(a);

 // lie group logarithm
 auto g_log = g.log();

 // lie algebra hat and vee maps
 auto a_hat = smooth::SO3d::hat(a);
 auto a_hat_vee = smooth::SO3d::vee(a_hat);

 // group adjoint
 auto Ad_g = g.Ad();

 // lie algebra adjoint
 auto ad_a = smooth::SO3d::ad(a);

 // derivatives of the exponential map
 auto dr_exp_v = smooth::SO3d::dr_exp(a);   // right derivative
 auto dl_exp_v = smooth::SO3d::dl_exp(a);   // left derivative
 auto dr_expinv_v = smooth::SO3d::dr_expinv(a);   // inverse of right derivative
 auto dl_expinv_v = smooth::SO3d::dl_expinv(a);   // inverse of left derivative

 // group action
 Eigen::Vector3d v = Eigen::Vector3d::Random();
 auto v_transformed = g * v;

 // memory mapping using Eigen::Map
 std::array<double, smooth::SO3d::RepSize> mem;
 Eigen::Map<const smooth::SO3d> m_g(mem.data());
 ```


## Concepts and Types

These [C++20 concepts](https://en.cppreference.com/w/cpp/concepts) are implemented in `concepts.hpp`.

* `Manifold`: type that supports `operator+()` (geodesic addition) and `operator-()` (geodesic subtraction)
  * Example: `smooth::ManifoldVector<Manifold>` from `manifold_vector.hpp`---a type to facilitate optimization over a dynamic number of `Manifold`s
  * Example: `Eigen::Vector3d`
  * Example: `Eigen::VectorXf`

* `LieGroup`: a `Manifold` that also implements Lie group operations (`exp`, `log`, `Ad`, etc...)
  * Example: `smooth::SO3<float>`
  * Example: `smooth::Bundle<LieGroup | Eigen::Matrix<Scalar, N, 1> ...>`


## Algorithms

Algorithms for `LieGroup` types can be used on regular Euclidean spaces via the `smooth::Tn` type.

### Tangent space differentiation

Available for `Manifold` types, see diff.hpp.

Supported techniques (see smooth::diff::Type):
* Numerical derivatives (default)
* Automatic differentiation using `autodiff` (must #include <smooth/compat/autodiff.hpp>)
* Automatic differentiation using Ceres 2.x (must #include <smooth/compat/ceres.hpp>)

Example: calculate ![](https://latex.codecogs.com/png.latex?\mathrm{d}^r&space;(\log(g_1&space;\circ&space;g_2))_{g_i}) for i=1, 2

```cpp
#include <smooth/diff.hpp>
#include <smooth/so3.hpp>

auto f = [](auto v1, auto v2) { return (v1 * v2).log(); };

smooth::SO3d g1 = smooth::SO3d::Random();
smooth::SO3d g2 = smooth::SO3d::Random();

// differentiate f at (g1, g2) w.r.t. first argument
auto [fval1, J1] = smooth::diff::dr(std::bind(f, std::placeholders::_1, g2), smooth::wrt(g1));

// differentiate f at (g1, g2) w.r.t. second argument
auto [fval2, J2] = smooth::diff::dr(std::bind(f, g1, std::placeholders::_1), smooth::wrt(g2));

// differentiate f at (g1, g2) w.r.t. both arguments
auto [fval, J] = smooth::diff::dr(f, smooth::wrt(g1, g2));

// Now J == [J1, J2]
```

### Non-linear least squares optimization

Available for `Manifold` types, see optim.hpp.

The minimize() function implements a Levenberg-Marquardt trust-region procedure to find
a local minimum.  All derivatives and computations are done in the tangent space as opposed
to e.g. Ceres which uses derivatives w.r.t. the parameterization.

A sparse solver is implemented, but it is currently only available when analytical
derivatives are provided.

Example: Calculate ![](https://latex.codecogs.com/png.latex?\mathrm{argmin}_{g_1}&space;\\|\log(g_1&space;\circ&space;g_2)\\|_2^2)

```cpp
#include <smooth/optim.hpp>
#include <smooth/so3.hpp>

// function defining residual
auto f = [](auto v1, auto v2) { return (v1 * v2).log(); };

smooth::SO3d g1 = smooth::SO3d::Random();
const smooth::SO3d g2 = smooth::SO3d::Random();

// minimize || f ||^2 w.r.t. first argument (g1 is modified in-place)
smooth::minimize(std::bind(f, std::placeholders::_1, g2), smooth::wrt(g1));

// Now g1 == g2.inverse()
```

### Bezier curve evaluation and fitting

Available for `LieGroup` types, see spline/bezier.hpp.

Bezier splines are piecewise defined via
[Bernstein polynomials](https://en.wikipedia.org/wiki/Bernstein_polynomial) and pass through
the control points. See examples/spline_fit.cpp for usage.


### B-spline evaluation and fitting

Available for `LieGroup` types, see spline/bspline.hpp.

B-splines have local support and generally do not pass through the control points.
See examples/bspline.cpp and examples/spline_fit.cpp for usage.


## Compatibility

Utility headers for interfacing with adjacent software are included.

* compat/autodiff.hpp: Use the [autodiff](https://autodiff.github.io/) library as a back-end for automatic differentiation
* compat/ceres.hpp: Local parameterization for [Ceres](http://ceres-solver.org/) on-manifold optimization, and use the Ceres automatic differentiation as a back-end
* compat/odeint.hpp: Numerical integration using [`boost::odeint`](https://www.boost.org/doc/libs/1_76_0/libs/numeric/odeint/doc/html/index.html)
* compat/ros.hpp: Memory mapping of [ROS/ROS2](https://www.ros.org/) message types


## Related Projects

Two projects that have served as inspiration for `smooth` are [`manif`][manif-link]---which
also has an accompanying [paper][manif-paper-link] that is a great practical introduction to
Lie theory---and [`Sophus`][sophus-link]. Certain design decisions are different in
`smooth`: derivatives are with respect to tangent elements as in `manif`, but the tangent
types are Eigen vectors like in `Sophus`. This library also includes the Bundle type which
facilitates control and estimation tasks, as well as utilities such as differentiation,
optimization, and splines. Finally `smooth` is written in C++20 and leverages modern
features such as [concepts](https://en.cppreference.com/w/cpp/language/constraints) and
[ranges](https://en.cppreference.com/w/cpp/ranges).


<!-- MARKDOWN LINKS AND IMAGES -->
[doc-link]: https://pettni.github.io/smooth

[ci-shield]: https://img.shields.io/github/workflow/status/pettni/smooth/build_and_test/master?style=flat-square
[ci-link]: https://github.com/pettni/lie/actions/workflows/build_and_test.yml

[cov-shield]: https://img.shields.io/codecov/c/gh/pettni/smooth/master?style=flat-square
[cov-link]: https://codecov.io/gh/pettni/smooth

[license-shield]: https://img.shields.io/github/license/pettni/smooth.svg?style=flat-square
[license-link]: https://github.com/pettni/smooth/blob/master/LICENSE

[manif-link]: https://github.com/artivis/manif/
[manif-paper-link]: https://arxiv.org/abs/1812.01537
[sophus-link]: https://github.com/strasdat/Sophus/

