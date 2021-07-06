# smooth: Lie Theory for Robotics

[![CI Build and Test][ci-shield]][ci-link]
[![Code coverage][cov-shield]][cov-link]
[![Documentation][doc-shield]][doc-link]
[![License][license-shield]][license-link]

<img src="media/ode.png" width="300">  <img src="media/bspline.png" width="300">

Do you want to do things like this on a Lie group?

 * Algebraic manipulation and analytic tangent space derivatives (example code below)
 * Numerical integration (left figure shows the solution of an ODE on SO(3) x R(3), see `examples/odeint.cpp`)
 * Automatic differentiation
 * Optimization
 * Interpolation (right figure shows a B-spline of order 5 on SO(3), see `examples/bspline.cpp`)

Then this project may be of interest. **Currently in development**, the goal is to
facilitate the use of Lie theory for robotics practitioners.

The following common Lie groups are implemented:
 * SO(2) with complex number (S(1)) memory representation
 * SO(3) with quaternion (S(3)) memory representation
 * SE(2)
 * SE(3)
 * A Bundle type to treat Lie group products G = G\_1 x ... x G\_n as a single Lie group. The Bundle type also supports R(n) components as Eigen vectors

These additional groups may or may not be implemented in the future:
 * The "IMU group" SE\_2(3)
 * Orthogonal matrices of any dimension: SO(n)
 * Unitary matrices of any dimension: SU(n)

The guiding principles for `smooth` are **brevity, reliability and compatability**.

*Since the project is currently under development, breaking changes should be expected.*


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

* ```Manifold```: type that supports ```operator+()```, ```operator-()```
  * Example: ```smooth::ManifoldVector<Manifold>``` from ```manifold_vector.hpp``` which is convenient for optimizaiton over dynamic numbers of ```Manifold```s
  * Algorithms: Differentiation, optimization, averaging

* ```RnLike```: a ```Manifold``` that is an ```Eigen``` vector type
  * Example: ```Eigen::VectorXd```

* ```StaticRnLike```: an ```RnLike``` whose size is known at compile-time
  * Example: ```Eigen::Vector3d```

* ```LieGroup```: a ```Manifold``` that also supports Lie group operations
  * Example: ```smooth::SO3d```
  * Example: ```smooth::Bundle<LieGroupLike ...>```
  * Algorithms: Numerical integration

* ```LieGroupLike```: a type that is either ```StaticRnLike``` or a ```LieGroup```
  * Algorithms: Splines


## Algorithms

Available:

* Tangent space differentiation on ```Manifold``` (`diff.hpp`)
* Non-linear least squares optimization on ```Manifold``` (`nls.hpp`)
* Bezier curve evaluation and fitting on ```LieGroupLike```  (`interp/bezier.hpp`)
* B-spline evaluation and fitting on ```LieGroupLike``` (`interp/bspline.hpp`)

Planned:

* Average on ```Manifold```


## Compatibility

Utility headers for interfacing with adjacent software are provided in `smooth/compat`

* Automatic differentiation in tangent space using [autodiff](https://autodiff.github.io/) or [Ceres](http://ceres-solver.org)
* Zero-copy memory mapping of [ROS/ROS2](https://www.ros.org/) message types
* Local parameterization for [Ceres](http://ceres-solver.org/) on-manifold optimization
* Numerical integration using [`boost::odeint`](https://www.boost.org/doc/libs/1_76_0/libs/numeric/odeint/doc/html/index.html)


## Related projects

Two similar projects that have served as inspiration for `smooth` are [`manif`](https://github.com/artivis/manif/), which also has an accompanying paper, and [`Sophus`](https://github.com/strasdat/Sophus/). Certain design decisions are different in `smooth`: jacobians are with respect to the tangent space as in `manif`, but the tangent types are Eigen vectors like in `Sophus`. This library also includes the Bundle type which greatly facilitates control and estimation tasks, and is written in C++20 which enables cleaner code as well as saner compiler error messages.

<!-- MARKDOWN LINKS AND IMAGES -->
[ci-shield]: https://img.shields.io/github/workflow/status/pettni/smooth/build_and_test/master?style=flat-square
[ci-link]: https://github.com/pettni/lie/actions/workflows/build_and_test.yml

[cov-shield]: https://img.shields.io/codecov/c/gh/pettni/smooth/master?style=flat-square
[cov-link]: https://codecov.io/gh/pettni/smooth

[doc-shield]: https://img.shields.io/static/v1?label=&message=Documentation&color=orange&style=flat-square
[doc-link]: https://pettni.github.io/smooth

[license-shield]: https://img.shields.io/github/license/pettni/smooth.svg?style=flat-square
[license-link]: https://github.com/pettni/smooth/blob/master/LICENSE

[license-shield]: https://img.shields.io/github/license/pettni/smooth.svg?style=flat-square
[license-link]: https://github.com/pettni/smooth/blob/master/LICENSE

