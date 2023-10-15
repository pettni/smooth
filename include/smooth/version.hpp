#pragma once

#define SMOOTH_VERSION v1_0

#ifndef SMOOTH_BEGIN_NAMESPACE
#define SMOOTH_BEGIN_NAMESPACE \
  namespace smooth {           \
  inline namespace SMOOTH_VERSION {
#define SMOOTH_END_NAMESPACE \
  }                          \
  }
#endif
