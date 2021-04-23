# smooth: A C++20 Lie Theory Library and Book for Robotics

# Roadmap

## Book todos

- [x] ceres derivatives
- [ ] reorganize dynamics: bring in system linearization and Magnus expansion
- Splines:
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
    - [ ] fit bsplines
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
 - [ ] get rid of large random header (use eigen seed)
 - [ ] make tests pass float in release mode
