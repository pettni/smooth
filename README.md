# smooth: Lie Theory for Robotics (C++20 library and book)

[![build_and_test](https://github.com/pettni/lie/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pettni/lie/actions/workflows/build_and_test.yml) [![codecov](https://codecov.io/gh/pettni/lie/branch/master/graph/badge.svg?token=M2S4HO9ZIG)](https://codecov.io/gh/pettni/lie)

<img src="/media/ode.png" width="300">  <img src="/media/bspline.png" width="300">

Do you want to do any of the following on a Lie group (or learn the theory)?

 * [x] Algebraic manipulation and analytic tangent space derivatives (example code below)
 * [x] Numerical integration (left figure shows the solution of an ODE on SO(3) x R(3), see `examples/odeint.cpp`)
 * [x] Automatic differentiation
 * [x] Optimization
 * [x] Interpolation (right figure shows a B-spline of order 5 on SO(3), see `examples/bspline.cpp`)

Then this project may be of interest. **Currently in development**, the goal is to
facilitate the use of Lie theory for robotics practitioners.

The following common lie groups are implemented:
 * SO(2) with complex number (S(1)) memory representation
 * SO(3) with quaternion (S(3)) memory representation
 * SE(2)
 * SE(3)
 * A Bundle type to treat Lie group products G = G\_1 x ... x G\_n as a single Lie group. The Bundle type also supports R(n) components as Eigen vectors

These additional groups may or may not be implemented in the future:
  - The "IMU group" SE\_2(3)
  - Orthogonal matrices of any dimension: SO(n)
  - Unitary matrices of any dimension: SU(n)
  - A "dynamic collection" type that exposes the Lie group interface for an `std::ranges::range` container

The guiding principles for `smooth` are **brevity, reliability and compatability**. The end goal is a **book** that describes theory and algorithms, and a **library** with high-quality implementations, and with a strong mapping between the two so that the book serves as a manual of the library.

*Since the project is currently under development, expect poor structure in the book, and API changes in the library.*


## Group algebra examples

 ```
 using G = smooth::SO3d;    // or SO2d, SE2d, SE3d, Bundle<SO3d, T3d> etc...
 using Tangent = typename G::Tangent;

 // construct a random group element and a random tangent element
 G g = G::Random();
 Tangent a = Tangent::Random();

 // lie group exponential
 auto exp_a = G::exp(a);

 // lie group logarithm
 auto g_log = g.log();

 // lie algebra hat and vee maps
 auto a_hat = G::hat(a);
 auto a_hat_vee = G::vee(a_hat);

 // group adjoint
 auto Ad_g = g.Ad();

 // lie algebra adjoint
 auto ad_a = G::ad(a);

 // derivatives of the exponential map
 auto dr_exp_v = G::dr_exp(a);   // right derivative
 auto dl_exp_v = G::dl_exp(a);   // left derivative
 auto dr_expinv_v = G::dr_expinv(a);   // inverse of right derivative
 auto dl_expinv_v = G::dl_expinv(a);   // inverse of left derivative

 // group action
 typename G::Vector v = G::Vector::Random();
 auto v_trans = g * v;

 // memory mapping using Eigen::Map
 std::array<double, G::lie_size> mem;
 Eigen::Map<const G> m_g(mem.data());
 ```


## Algorithms

Available:

* [x] Tangent space differentiation (`diff.hpp`)
* [x] Bezier curve evaluation and fitting (`interp/bezier.hpp`)
* [x] B-spline evaluation and fitting (`interp/bspline.hpp`)
* [x] Non-linear least squares optimization (`nls.hpp`)

Planned:

* [ ] Lie group means (`mean.hpp`)

Algorithms can also be made to work with regular ```Eigen``` types via the Bundle type.


## Compatibility

Utility headers for interfacing with adjacent software are provided in `smooth/compat`

* [x] Automatic differentiation in tangent space using [autodiff](https://autodiff.github.io/)
* [x] Zero-copy memory mapping of [ROS/ROS2](https://www.ros.org/) message types
* [x] Local parameterization for [Ceres](http://ceres-solver.org/) on-manifold optimization
* [x] Numerical integration using [`boost::odeint`](https://www.boost.org/doc/libs/1_76_0/libs/numeric/odeint/doc/html/index.html)


## Related projects

Two similar projects that have served as inspiration for `smooth` are [`manif`](https://github.com/artivis/manif/), which also has an accompanying paper, and [`Sophus`](https://github.com/strasdat/Sophus/). Certain design decisions are different in `smooth`: jacobians are with respect to the tangent space as in `manif`, but the tangent types are Eigen vectors like in `Sophus`. This library also includes the Bundle type which greatly facilitates control and estimation tasks, and is written in C++20 which enables cleaner code as well as saner compiler error messages.


# Next steps

## Book

- [x] reorganize dynamics: bring in system linearization and Magnus expansion
- [ ] clean up parameterization vs matrix groups stuff, use consistent X / x
- Work out missing parts
  - [ ] Probability theory
  - [/] Equivariance
  - [ ] Numerical Integration
  - [ ] Estimation
- Make readable
  - Part 1:
    - [ ] Introduction
    - [ ] Lie Groups
    - [ ] Lie Algebras
    - [ ] Exponential map
    - [/] Derivatives
    - [ ] Dynamical Systems
    - [ ] Equivariance
  - Part 2:
    - [x] Classical groups
    - [x] SO2
    - [x] SO3
    - [/] SE2
    - [/] SE3
  - Part 3:
    - [ ] Numerical integration
    - [ ] Control
    - [ ] Estimation
    - [ ] NLS
    - [ ] PGO / marginalization
    - [ ] Splines


## Library

- [ ] Better sparse solvers
  - Eigen SparseQR seems crappy for bspline fitting
  1. Use SPQR from suitesparse in current framework
  2. Use LDLT to solve full problem J' J x = -J' r
     - do this directly with a given lambda = 1 / D
     - do this inside of lmpar...
- [ ] pass options in NLS
- [ ] ceres autodiff

