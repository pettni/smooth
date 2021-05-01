# smooth: Lie Theory C++20 Library and Book for Robotics

Do you need to do any of the following on a Lie group manifold?

 * [x] Algebraic manipulation and analytic derivatives on common Lie groups
 * [x] Numerical integration
 * [ ] Automatic differentiation
 * [ ] Optimization
 * [x] Interpolation with splines
 * [ ] Design controllers
 * [ ] Design estimators

Then this project may be of interest. **Currently in development**, the goal is to
facilitate the use of Lie theory for robotics practitioners.

The following common lie groups are available:
  - SO(2)
  - SO(3)
  - SE(2)
  - SE(3)
  - A Bundle type to treat Lie group products `G = G_1 \times \ldots \times G_n` as a single Lie group

The end goal is a book that describes theory and algorithms, and a library with implementations, with a strong mapping between the two so that the book serves as a manual of the library.

Utility headers for interfacing with adjacent software are also included

 * [x] Zero-copy memory mapping of [ROS/ROS2](https://www.ros.org/) message types
 * [x] Local parameterization for [Ceres](http://ceres-solver.org/) on-manifold optimization
 * [x] Numerical integration using [`boost::odeint`](https://www.boost.org/doc/libs/1_76_0/libs/numeric/odeint/doc/html/index.html)

*Since the project is currently under development, expect poor structure and explanations in the book, and API changes in the library.*

## Background

Lie groups such as SO(3), SE(2) and SE(3) are very useful abstractions in robotics,
but it is my impression that knowledge of Lie theory remains sparse in the robotics
community. In particular there is a shortage of material that presents concepts in
a way that accessible to most roboticists, and with a focus on appications rather than
theorem proving.

## Related projects

Two similar projects that have served as inspiration for `smooth` are [manif](https://github.com/artivis/manif/), which also has an accompagnying paper, and [Sophus](https://github.com/strasdat/Sophus/). `smooth` makes some different design decisions, jacobians are with respect to the tangent space as in `manif`, but the tangent types are Eigen vectors like in `Sophus`. This library also includes the Bundle type which greatly facilitates control and estimation tasks, and is written using modern C++20 which enables cleaner and shorter code as well as saner compiler error messages.


# Roadmap

## Notes todos

- [x] ceres derivatives
- [ ] Levenberg-Marquardt trust-region optimization
- [ ] reorganize dynamics: bring in system linearization and Magnus expansion
- Splines:
  - [x] Bsplines derivatives w.r.t. control points
  - [ ] Fitting of bsplines
  - [ ] Fitting of cubic splines: with and without given velocity

## Library todos

- [x] cpp20 concepts and crtp
- [x] set up lib structure
- [x] support different storage types
- compatability
  - [x] boost odeint: read about algebras
  - [ ] autodiff: tangent derivative of any manifold function
  - [x] ceres: local parameterization of any group
  - [x] map ros msgs as storage type
- groups
  - [x] so2
  - [x] so3
  - [x] se2
  - [x] se3
  - [x] bundle
- algos:
  - imu preintegration
  - splines:
    - [x] evaluate bsplines
    - [ ] fit bsplines: need ceres or own LM optimizer...
    - [ ] fit cubic splines with and without given velocity
- debt:
 - [x] get rid of modifiable unordered storage and coeffs_ordered
    - just don't expose map in if not supported by storage
 - [x] write own storages -- the eigen ones are ugly
 - [x] change SE2/3 constructors to take translation first
 - [x] Avoid duplication of boilerplate
   - [x] trait to change n:th template arg to use lie_group_base for bundle
   - [x] macro to define constructor boilerplate for each group
 - [x] revisit small angle approximations
 - [x] en: make bundle support vectors with usual semantics and leave it there
 - [x] bundle: typedef bundle with default storage
 - [x] get rid of large random header (use eigen seed)
 - [x] make tests pass float in release mode
