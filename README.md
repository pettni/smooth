# smooth: Lie Theory for Robotics

[![CI Build and Test][ci-shield]][ci-link]
[![Code coverage][cov-shield]][cov-link]
[![License][license-shield]][license-link]

<img src="media/ode.png" width="300">  <img src="media/bspline.png" width="300">

In robotics it is often convenient to work in non-Euclidean manifolds. [Lie groups](https://en.wikipedia.org/wiki/Lie_group) are a class of manifolds that due to their symmetry are easy to work with, and are also good models for many robotic systems. The objective of this header-only C++20 library is to make it easy to use Lie groups in robotics software, by enabling things such as:

 * Algebraic manipulation and analytic tangent space derivatives (example code below)
 * Numerical integration (left figure shows the solution of an ODE on SO(3) x R(3), see `examples/odeint.cpp`)
 * Automatic differentiation
 * Optimization
 * Interpolation (right figure shows a B-spline of order 5 on SO(3), see `examples/bspline.cpp`)

The following common Lie groups are implemented:
 * smooth::SO2 with complex number (S(1)) memory representation
 * smooth::SO3 with quaternion (S(3)) memory representation
 * smooth::SE2
 * smooth::SE3
 * A smooth::Bundle type to treat Lie group products G = G\_1 x ... x G\_n as a single Lie group. The Bundle type also supports R(n) components as Eigen vectors

These additional groups may or may not be implemented in the future:
 * The "IMU group" SE\_2(3)
 * Orthogonal matrices of any dimension: SO(n)
 * Unitary matrices of any dimension: SU(n)

The guiding principles for `smooth` are **brevity, reliability and compatability**.

*This project is currently being developed, breaking changes should be expected.*


## Getting started

### Download and Build

Clone the repository and install it
```zsh
git clone https://github.com/pettni/smooth.git
cd smooth
mkdir build && cd build

# Specify a C++20-compatible compiler if your default does not support C++20.
# Build tests and/or examples as desired.
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++-10 -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF
make -j8
sudo make install
```

Alternatively, if using ROS or ROS2 just clone `smooth` into a catkin/[colcon](https://colcon.readthedocs.io/en/released/) workspace source folder and build the
workspace with a compiler that supports C++20. Example with colcon:
```zsh
colcon build --cmake-args -DCMAKE_CXX_COMPILER=/usr/bin/g++-10
```

### Use with cmake

To utilize `smooth` in your own project, include this in your `CMakeLists.txt`
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
 std::array<double, smooth::SO3d::lie_size> mem;
 Eigen::Map<const smooth::SO3d> m_g(mem.data());
 ```


## Concepts and Types

These [C++20 concepts](https://en.cppreference.com/w/cpp/concepts) are implemented in `concepts.hpp`.

* `Manifold`: type that supports `operator+()`, `operator-()`
  * Example: `smooth::ManifoldVector<Manifold>` from `manifold_vector.hpp` which is convenient for optimizaiton over dynamic numbers of `Manifold`s

* `RnLike`: a `Manifold` that is an `Eigen` vector type
  * Example: `Eigen::VectorXd`

* `LieGroup`: a `Manifold` that also supports Lie group operations
  * Example: `smooth::SO3d`
  * Example: `smooth::Bundle<LieGroupLike ...>`

* `LieGroupLike`: a type for which `lie_traits` is specialized
  * Example: All `LieGroup`, all `RnLike` with size known at compile-time


## Algorithms

* Tangent space differentiation on `Manifold` (`diff.hpp`)
* Non-linear least squares optimization on `Manifold` (`optim.hpp`)
* Bezier curve evaluation and fitting on `LieGroupLike`  (`spline/bezier.hpp`)
* B-spline evaluation and fitting on `LieGroupLike` (`spline/bspline.hpp`)


## Compatibility

Utility headers for interfacing with adjacent software are provided in `smooth/compat`

* Automatic differentiation in tangent space using [autodiff](https://autodiff.github.io/) or [Ceres](http://ceres-solver.org)
* Zero-copy memory mapping of [ROS/ROS2](https://www.ros.org/) message types
* Local parameterization for [Ceres](http://ceres-solver.org/) on-manifold optimization
* Numerical integration using [`boost::odeint`](https://www.boost.org/doc/libs/1_76_0/libs/numeric/odeint/doc/html/index.html)


## Related projects

Two similar projects that have served as inspiration for `smooth` are [`manif`](https://github.com/artivis/manif/), which also has an accompanying paper, and [`Sophus`](https://github.com/strasdat/Sophus/). Certain design decisions are different in `smooth`: jacobians are with respect to the tangent space as in `manif`, but the tangent types are Eigen vectors like in `Sophus`. This library also includes the Bundle type which facilitates control and estimation tasks, as well as additional utilities such as differentiation, optimization, and splines. Finally `smooth` is written in C++20 and leverages modern features such as [concepts](https://en.cppreference.com/w/cpp/language/constraints) and [ranges](https://en.cppreference.com/w/cpp/ranges).

<!-- MARKDOWN LINKS AND IMAGES -->
[doc-link]: https://pettni.github.io/smooth

[ci-shield]: https://img.shields.io/github/workflow/status/pettni/smooth/build_and_test/master?style=flat-square
[ci-link]: https://github.com/pettni/lie/actions/workflows/build_and_test.yml

[cov-shield]: https://img.shields.io/codecov/c/gh/pettni/smooth/master?style=flat-square
[cov-link]: https://codecov.io/gh/pettni/smooth

[license-shield]: https://img.shields.io/github/license/pettni/smooth.svg?style=flat-square
[license-link]: https://github.com/pettni/smooth/blob/master/LICENSE

