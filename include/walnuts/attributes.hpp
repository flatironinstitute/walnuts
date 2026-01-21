#pragma once

#ifdef __has_attribute
/**
 * Forces function to not be inline eligible
 * https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-noinline-function-attribute
 **/
#ifndef WALNUTS_NO_INLINE_
  #if __has_attribute(noinline)
  #define WALNUTS_NO_INLINE_ __attribute__((noinline))
  #else
  #define WALNUTS_NO_INLINE_
  #endif
#endif

/**
 * Compiler will attempt to inline this function more
 * https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-always_005finline-function-attribute
 **/
#ifndef WALNUTS_ALWAYS_INLINE_
  #if __has_attribute(always_inline)
  #define WALNUTS_ALWAYS_INLINE_ __attribute__((always_inline)) inline
  #else
  #define WALNUTS_ALWAYS_INLINE_ inline
  #endif
#endif

/**
 * Compiler treats function as hot spot and optimizes more aggressively
 * https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-hot-function-attribute
 **/
#ifndef WALNUTS_HOT_
  #if __has_attribute(hot)
  #define WALNUTS_HOT_ __attribute__((hot))
  #else
  #define WALNUTS_HOT_
  #endif
#endif

/**
 * Place function on the cold path for the compiler. Useful for functions that
 *construct error messages etc.
 * https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-cold-function-attribute
 **/
#ifndef WALNUTS_COLD_
  #if __has_attribute(cold)
  #define WALNUTS_COLD_ __attribute__((cold))
  #else
  #define WALNUTS_COLD_
  #endif
#endif

/**
 * Compiler will attempt to inline everything inside of the function
 * https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-flatten-function-attribute
 **/
#ifndef WALNUTS_FLATTEN_
  #if __has_attribute(flatten)
  #define WALNUTS_FLATTEN_ __attribute__((flatten))
  #else
  #define WALNUTS_FLATTEN_
  #endif
#endif

/**
 * For defining a function as pure
 * https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-pure-function-attribute
 **/
#ifndef WALNUTS_PURE_
#if __has_attribute(pure)
#define WALNUTS_PURE_ __attribute__((pure))
#else
#define WALNUTS_PURE_
#endif
#endif

#ifndef likely
#define likely(x) __builtin_expect((x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect((x), 0)
#endif

#else // ifdef __has_attribute

#ifndef WALNUTS_NO_INLINE_
#define WALNUTS_NO_INLINE_
#endif

#ifndef WALNUTS_ALWAYS_INLINE_
#define WALNUTS_ALWAYS_INLINE_ inline
#endif

#ifndef WALNUTS_HOT_
#define WALNUTS_HOT_
#endif

#ifndef WALNUTS_COLD_
#define WALNUTS_COLD_
#endif

#ifndef WALNUTS_FLATTEN_
#define WALNUTS_FLATTEN_
#endif

#ifndef WALNUTS_PURE_
#define WALNUTS_PURE_
#endif

#ifndef likely
#define likely(x) __builtin_expect((x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect((x), 0)
#endif
#endif // ifdef __has_attribute
