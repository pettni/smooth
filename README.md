# lie

Lie theory for robotics: manual and library

# Roadmap

## Book todos

- [ ] reorganize dynamics
- Splines:
  - [ ] Fitting of bsplines
  - [ ] Fitting of cubic splines: with and without given velocity

## Library todos

- [x] cpp20 concepts and crtp
- [x] set up lib structure
- [x] support different storage types
- compatability
  - [ ] boost odeint: read about algebras
  - [ ] autodiff: tangent derivative of any manifold function
  - [ ] ceres: local parameterization of any group
  - [ ] map ros msgs as storage type
- groups
  - [ ] en (inherits from Eigen vec)
  - [x] so2
  - [x] so3
  - [x] se2
  - [x] se3
  - [ ] bundle
- algos:
  - splines:
    - [ ] evaluate bsplines
    - [ ] fit bsplines
    - [ ] for cubic splines with and without given velocity
- debt:
 - [ ] change SE2/3 constructors to take translation first
 - [ ] do small angle approximations

### Design choices

 - Return auto to avoid evaluating Eigen temp expressions?
 - Get rid of unordered storage support?
