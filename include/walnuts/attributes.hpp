#pragma once
#ifndef WALNUTS_NO_INLINE_
#define WALNUTS_NO_INLINE_ __attribute__((noinline))
#endif

#ifndef WALNUTS_ALWAYS_INLINE_
#define WALNUTS_ALWAYS_INLINE_ __attribute__((always_inline)) inline
#endif

#ifndef WALNUTS_HOT_
#define WALNUTS_HOT_ __attribute__((hot))
#endif

#ifndef WALNUTS_COLD_
#define WALNUTS_COLD_ __attribute__((cold))
#endif

#ifndef WALNUTS_FLATTEN_
#define WALNUTS_FLATTEN_ __attribute__((flatten))
#endif

#ifndef WALNUTS_PURE_
#define WALNUTS_PURE_ __attribute__((pure))
#endif

#ifndef likely
#define likely(x) __builtin_expect((x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect((x), 0)
#endif
