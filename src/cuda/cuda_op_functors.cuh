#ifndef cuda_op_functors_CUH
#define cuda_op_functors_CUH

#include <cuda_runtime.h>
#include <cmath>
#include <math_constants.h>
#include <climits>
#include <type_traits>
#include <limits>

template <typename T>
struct AddOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return a + b; }
};

template <typename T>
struct SubOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return a - b; }
};

template <typename T>
struct MulOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return a * b; }
};

template <typename T>
struct DivOpT
{
    static __device__ __forceinline__ T apply(T a, T b)
    {
        if constexpr (std::is_integral<T>::value)
            return b == 0 ? static_cast<T>(0) : a / b;
        else
            return a / b;
    }
};

template <typename T>
struct PowOpT
{
    static __device__ __forceinline__ T int_pow(T base, int exp) {
        T res = (T)1;
        while (exp > 0) {
            if (exp & 1) res *= base;
            base *= base;
            exp >>= 1;
        }
        return res;
    }

    static __device__ __forceinline__ T apply(T a, T b)
    {
        bool is_integer_exp = false;
        
        if constexpr (std::is_floating_point<T>::value) {
            if (b == floor(b)) is_integer_exp = true;
        } else {
            is_integer_exp = true;
        }

        if (is_integer_exp) {
            int exp = static_cast<int>(b);
            
            if (exp == 0) return (T)1;
            if (exp == 1) return a;
            if (exp == 2) return a * a;
            if (exp == 3) return a * a * a;

            if (exp < 0) {
                if constexpr (std::is_floating_point<T>::value) {
                    return (T)1.0 / int_pow(a, -exp);
                } else {
                    return (T)0; 
                }
            }
            return int_pow(a, exp);
        }

        if constexpr (std::is_same<T, float>::value) {
            return powf(a, b);
        } else if constexpr (std::is_same<T, double>::value) {
            return pow(a, b);
        } else {
            return static_cast<T>(powf(static_cast<float>(a), static_cast<float>(b)));
        }
    }
};

template <typename T>
struct GTOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return (a > b) ? static_cast<T>(1) : static_cast<T>(0); }
};

template <typename T>
struct LTOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return (a < b) ? static_cast<T>(1) : static_cast<T>(0); }
};

template <typename T>
struct EQOpT
{
    static __device__ __forceinline__ T apply(T a, T b)
    {
        if constexpr (std::is_floating_point<T>::value)
        {
            const T epsilon = static_cast<T>(1e-6);
            return (fabs(a - b) < epsilon) ? static_cast<T>(1) : static_cast<T>(0);
        }
        else
        {
            return (a == b) ? static_cast<T>(1) : static_cast<T>(0);
        }
    }
};

template <typename T>
struct NEOpT
{
    static __device__ __forceinline__ T apply(T a, T b)
    {
        if constexpr (std::is_floating_point<T>::value)
        {
            const T epsilon = static_cast<T>(1e-6);
            return (fabs(a - b) >= epsilon) ? static_cast<T>(1) : static_cast<T>(0);
        }
        else
        {
            return (a != b) ? static_cast<T>(1) : static_cast<T>(0);
        }
    }
};

template <typename T>
struct GEOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return (a >= b) ? static_cast<T>(1) : static_cast<T>(0); }
};

template <typename T>
struct LEOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return (a <= b) ? static_cast<T>(1) : static_cast<T>(0); }
};

template <typename T>
struct MaxOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return (a > b) ? a : b; }
};

template <typename T>
struct MinOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return (a < b) ? a : b; }
};

template <typename T>
struct ExpOpT
{
    static __device__ __forceinline__ T apply(T a)
    {
        if constexpr (std::is_same<T, float>::value)
            return expf(a);
        else if constexpr (std::is_same<T, double>::value)
            return exp(a);
        else
            return static_cast<T>(expf(static_cast<float>(a)));
    }
};

template <typename T>
struct LogOpT
{
    static __device__ __forceinline__ T apply(T a)
    {
        if constexpr (std::is_same<T, float>::value)
            return logf(a);
        else if constexpr (std::is_same<T, double>::value)
            return log(a);
        else
            return static_cast<T>(logf(static_cast<float>(a)));
    }
};

template <typename T>
struct SqrtOpT
{
    static __device__ __forceinline__ T apply(T a)
    {
        if constexpr (std::is_same<T, float>::value)
            return sqrtf(a);
        else if constexpr (std::is_same<T, double>::value)
            return sqrt(a);
        else
            return static_cast<T>(sqrtf(static_cast<float>(a)));
    }
};

template <typename T>
struct AbsOpT
{
    static __device__ __forceinline__ T apply(T a)
    {
        if constexpr (std::is_unsigned<T>::value)
            return a;
        else
            return (a < static_cast<T>(0)) ? -a : a;
    }
};

template <typename T>
struct NegOpT
{
    static __device__ __forceinline__ T apply(T a) { return -a; }
};

template <typename T>
struct FloorOpT
{
    static __device__ __forceinline__ T apply(T a)
    {
        if constexpr (std::is_same<T, float>::value)
            return floorf(a);
        else if constexpr (std::is_same<T, double>::value)
            return floor(a);
        else
            return a;
    }
};

template <typename T>
struct CeilOpT
{
    static __device__ __forceinline__ T apply(T a)
    {
        if constexpr (std::is_same<T, float>::value)
            return ceilf(a);
        else if constexpr (std::is_same<T, double>::value)
            return ceil(a);
        else
            return a;
    }
};

template <typename T>
struct RoundOpT
{
    static __device__ __forceinline__ T apply(T a)
    {
        if constexpr (std::is_same<T, float>::value)
            return roundf(a);
        else if constexpr (std::is_same<T, double>::value)
            return round(a);
        else
            return a;
    }
};

template <typename T>
struct SinOpT
{
    static __device__ __forceinline__ T apply(T a)
    {
        if constexpr (std::is_same<T, float>::value)
            return sinf(a);
        else if constexpr (std::is_same<T, double>::value)
            return sin(a);
        else
            return static_cast<T>(sinf(static_cast<float>(a)));
    }
};

template <typename T>
struct CosOpT
{
    static __device__ __forceinline__ T apply(T a)
    {
        if constexpr (std::is_same<T, float>::value)
            return cosf(a);
        else if constexpr (std::is_same<T, double>::value)
            return cos(a);
        else
            return static_cast<T>(cosf(static_cast<float>(a)));
    }
};

template <typename T>
struct TanOpT
{
    static __device__ __forceinline__ T apply(T a)
    {
        if constexpr (std::is_same<T, float>::value)
            return tanf(a);
        else if constexpr (std::is_same<T, double>::value)
            return tan(a);
        else
            return static_cast<T>(tanf(static_cast<float>(a)));
    }
};

template <typename T>
struct ArgMaxOpT
{
    static __device__ __forceinline__ bool apply(T a, T b) { return a > b; }
};

template <typename T>
struct ArgMinOpT
{
    static __device__ __forceinline__ bool apply(T a, T b) { return a < b; }
};

template <typename T, typename OpT>
struct ArgIdentity
{
    __device__ __forceinline__ static T get_init_val() { return static_cast<T>(0); }
};

template <typename T>
struct ArgIdentity<T, MulOpT<T>>
{
    __device__ __forceinline__ static T get_init_val() { return static_cast<T>(1); }
};

template <typename T>
struct ArgIdentity<T, AddOpT<T>>
{
    __device__ __forceinline__ static T get_init_val() { return static_cast<T>(0); }
};

template <typename T>
struct ArgIdentity<T, MaxOpT<T>>
{
    static constexpr T integral_value = std::numeric_limits<T>::lowest();
    __device__ __forceinline__ static T get_init_val()
    {
        if constexpr (std::is_floating_point<T>::value)
        {
            return (T)-CUDART_INF_F;
        }
        else if constexpr (std::is_same<T, bool>::value)
        {
            return false;
        }
        else
        {
            return integral_value;
        }
    }
};

template <typename T>
struct ArgIdentity<T, MinOpT<T>>
{
    static constexpr T integral_value = std::numeric_limits<T>::max();
    __device__ __forceinline__ static T get_init_val()
    {
        if constexpr (std::is_floating_point<T>::value)
        {
            return (T)CUDART_INF_F;
        }
        else if constexpr (std::is_same<T, bool>::value)
        {
            return true;
        }
        else
        {
            return integral_value;
        }
    }
};

template <typename T>
struct ArgIdentity<T, ArgMaxOpT<T>> : ArgIdentity<T, MaxOpT<T>>
{
};

template <typename T>
struct ArgIdentity<T, ArgMinOpT<T>> : ArgIdentity<T, MinOpT<T>>
{
};

#endif