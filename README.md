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
  - [x] so2
  - [x] so3
  - [x] se2
  - [x] se3
  - [x] bundle
- algos:
  - splines:
    - [ ] evaluate bsplines
    - [ ] fit bsplines
    - [ ] for cubic splines with and without given velocity
- debt:
 - [x] get rid of modifiable unordered storage and coeffs_ordered
    - just don't expose map in if not supported by storage
 - [x] write own storages -- the eigen ones are ugly
 - [x] change SE2/3 constructors to take translation first
 - [ ] trait to change n:th template arg
 - [ ] inject boilerplate with macro and remove lie_group_base
 - [ ] do small angle approximations
 - [x] en: make bundle support vectors with usual semantics and leave it there
 - [x] bundle: typedef bundle with default storage


### Design choices

 - Return auto to avoid evaluating Eigen temp expressions?
 - Get rid of unordered storage support?
 - Use CRTP or nah?
 - En
   - Same approach as rest of lib
     - Pro: more consistent
     - Pro: stronger typed
     - Pro: may be able to make something clever for semi-simple group products
     - Con: maybe not as fast
     - Con: more code
     - Con: must use translation() or similar in syntax
   - [x] No En type, do if constexpr in Bundle
     - Pro: minimal code
     - Con: bundle becomes more complicated
