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
  - [ ] en (plugin to MatrixBase)
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
 - En
   - Inherit from matrix
     - Pro: faster because of template expressions
     - Con: can not use LieGroupBase
   - Inject API into matrixbase
     - Pro: less code
     - Pro: fluid syntax
     - Pro: faster because of template expressions
     - Con: must include plugin before any eigen header
     - ~~Con: must re-name methods~~
     - Con: must remove storage concept
     - Con: can not use LieGroupBase
     - Con: operator* ambiguous
     - Con: imposes constraints on include order
   - Same approach as rest of lib
     - Pro: more consistent
     - Pro: stronger typed
     - Con: maybe not as fast
     - Con: more code
     - Con: must use translation() or similar in syntax
   - No En type, do if constexpr in Bundle
     - Pro: minimal code
     - Con: bundle becomes more complicated
