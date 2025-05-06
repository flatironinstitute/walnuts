#ifndef INCLUDE_NUTS_MACROS_HPP
#define INCLUDE_NUTS_MACROS_HPP

namespace nuts {

#ifdef __GNUC__

/**
 * If statements predicate tagged with this attribute are expected to
 * be true most of the time. This effects inlining decisions.
 */
#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif
// 1
#ifdef __has_attribute
// 2
#if __has_attribute(noinline) && __has_attribute(cold)
// 3
#ifndef NUTS_COLD_PATH
/**
 * Functions tagged with this attribute are not inlined and moved
 *  to a cold branch to tell the CPU to not attempt to pre-fetch
 *  the associated function.
 */
#define NUTS_COLD_PATH __attribute__((noinline, cold))
// 3
#endif
// 2
#endif
// 1
#endif

// 1
#ifndef NUTS_COLD_PATH
#define NUTS_COLD_PATH
// 1
#endif

// 1
#ifndef NUTS_NO_INLINE
#define NUTS_NO_INLINE __attribute__((noinline))
// 1
#endif

// 1
#ifndef NUTS_ALWAYS_INLINE
#define NUTS_ALWAYS_INLINE __attribute__((always_inline)) inline
// 1
#endif

/**
 * Functions tagged with this attribute are pure functions, i.e. they
 * do not modify any global state and only depend on their input arguments.
 */
#ifndef NUTS_PURE
#define NUTS_PURE __attribute__((pure))
#endif

#else
#define likely(x) (x)
#define unlikely(x) (x)
#define NUTS_COLD_PATH
#define NUTS_NO_INLINE
#define NUTS_ALWAYS_INLINE inline
#define NUTS_PURE
#endif

}  // namespace nuts

#endif
