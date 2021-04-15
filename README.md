# lie

Lie theory for robotics: manual and library

# Roadmap

## Book

- Comprehensive theory and algorithms
- Code snippets that are CI-tested
- Splines:
  - Fitting of bsplines
  - Fitting of cubic splines: with and without given velocity

## Library

- cpp20 concepts and crtp
- storage options: std::array, raw ptr, ros msg
- compatability
  - [ ] boost odeint
  - [ ] eigen map ros msgs
  - [ ] autodiff
  - [ ] ceres
- groups
  - [ ] en (inherits from Eigen vec)
  - [x] so2
  - [x] so3
  - [ ] se2
  - [ ] se3
  - [ ] bundle
- algos:
  - splines:
    - [ ] evaluate bsplines
    - [ ] fit bsplines
    - [ ] fir cubic splines with and without given velocity

# Next steps

- [ ] reorganize dynamics in book
- [x] set up lib structure
- [x] support different storage types

