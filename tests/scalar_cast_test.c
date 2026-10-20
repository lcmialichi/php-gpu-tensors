#include "php.h"
#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "../src/data_types.c"

static void check_integer_promotion(void)
{
    const dtype_t types[] = {
        DTYPE_INT8, DTYPE_INT16, DTYPE_INT32, DTYPE_INT64,
        DTYPE_UINT8, DTYPE_UINT16, DTYPE_UINT32, DTYPE_UINT64
    };
    const dtype_t expected[8][8] = {
        {DTYPE_INT8, DTYPE_INT16, DTYPE_INT32, DTYPE_INT64, DTYPE_INT16, DTYPE_INT32, DTYPE_INT64, DTYPE_UNKNOWN},
        {DTYPE_INT16, DTYPE_INT16, DTYPE_INT32, DTYPE_INT64, DTYPE_INT16, DTYPE_INT32, DTYPE_INT64, DTYPE_UNKNOWN},
        {DTYPE_INT32, DTYPE_INT32, DTYPE_INT32, DTYPE_INT64, DTYPE_INT32, DTYPE_INT32, DTYPE_INT64, DTYPE_UNKNOWN},
        {DTYPE_INT64, DTYPE_INT64, DTYPE_INT64, DTYPE_INT64, DTYPE_INT64, DTYPE_INT64, DTYPE_INT64, DTYPE_UNKNOWN},
        {DTYPE_INT16, DTYPE_INT16, DTYPE_INT32, DTYPE_INT64, DTYPE_UINT8, DTYPE_UINT16, DTYPE_UINT32, DTYPE_UINT64},
        {DTYPE_INT32, DTYPE_INT32, DTYPE_INT32, DTYPE_INT64, DTYPE_UINT16, DTYPE_UINT16, DTYPE_UINT32, DTYPE_UINT64},
        {DTYPE_INT64, DTYPE_INT64, DTYPE_INT64, DTYPE_INT64, DTYPE_UINT32, DTYPE_UINT32, DTYPE_UINT32, DTYPE_UINT64},
        {DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UINT64, DTYPE_UINT64, DTYPE_UINT64, DTYPE_UINT64}
    };
    for (size_t a = 0; a < 8; a++) for (size_t b = 0; b < 8; b++)
    {
        dtype_t common = expected[a][b];
        assert(promote_types(types[a], types[b]) == common);
        assert(promote_types_for_arithmetic(types[a], types[b], OP_ADD) == common);
        assert(promote_types_for_comparison(types[a], types[b]) == common);
        if (common != DTYPE_UNKNOWN)
        {
            assert(can_safely_cast_to(types[a], common));
            assert(can_safely_cast_to(types[b], common));
        }
    }
    assert(promote_types(DTYPE_UNKNOWN, DTYPE_BOOL) == DTYPE_UNKNOWN);
    assert(promote_types((dtype_t)-1, DTYPE_INT64) == DTYPE_UNKNOWN);
    assert(promote_types_for_arithmetic(DTYPE_INT64, DTYPE_UINT64, OP_DIV) == DTYPE_UNKNOWN);
    assert(promote_types_for_arithmetic(DTYPE_INT64, DTYPE_INT64, OP_DIV) == DTYPE_FLOAT64);
    assert(promote_scalar_for_arithmetic(DTYPE_UINT64, DTYPE_INT64, OP_ADD, 0) == DTYPE_UNKNOWN);
    assert(promote_scalar_for_arithmetic(DTYPE_INT64, DTYPE_INT64, OP_DIV, 0) == DTYPE_FLOAT64);
    assert(promote_scalar_for_arithmetic(DTYPE_INT64, DTYPE_FLOAT64, OP_ADD, 0) == DTYPE_FLOAT64);
}

int main(void)
{
    check_integer_promotion();
    scalar_value_t input = {.v.i64 = 7, .dtype = DTYPE_INT64};
    scalar_value_t value = cast_single_value(input, DTYPE_INT16);
    assert(value.dtype == DTYPE_INT16);
    long double number;
    for (int source = 0; source < DTYPE_UNKNOWN; source++)
    {
        scalar_value_t typed = cast_single_value(input, (dtype_t)source);
        assert(typed.dtype == source);
        assert(scalar_value_as_long_double(typed, &number));
        assert(number == (source == DTYPE_BOOL ? 1 : 7));
        for (int target = 0; target < DTYPE_UNKNOWN; target++)
        {
            value = cast_single_value(typed, (dtype_t)target);
            assert(value.dtype == target);
            assert(scalar_value_as_long_double(value, &number));
            assert(number == (source == DTYPE_BOOL || target == DTYPE_BOOL ? 1 : 7));
        }
    }
    assert(!dtype_size((dtype_t)-1));
    assert(!can_safely_cast_to(DTYPE_UNKNOWN, DTYPE_UNKNOWN));
    assert(!can_safely_cast_to(DTYPE_BOOL, DTYPE_FLOAT16));
    assert(can_safely_cast_to(DTYPE_FLOAT16, DTYPE_FLOAT64));
    input.dtype = DTYPE_FLOAT64;
    input.v.f64 = NAN;
    for (int target = DTYPE_INT8; target <= DTYPE_UINT64; target++)
        assert(cast_single_value(input, (dtype_t)target).dtype == DTYPE_UNKNOWN);
    input.v.f64 = 0x1p63;
    assert(cast_single_value(input, DTYPE_INT64).dtype == DTYPE_UNKNOWN);
    input.v.f64 = 0x1p64;
    assert(cast_single_value(input, DTYPE_UINT64).dtype == DTYPE_UNKNOWN);
    input.v.f64 = -0.5;
    assert(cast_single_value(input, DTYPE_UINT8).dtype == DTYPE_UNKNOWN);
    input.v.f64 = -3.75;
    value = cast_single_value(input, DTYPE_INT16);
    assert(value.dtype == DTYPE_INT16 && value.v.i16 == -3 && value.is_neg);
    input.dtype = DTYPE_UINT64;
    input.v.u64 = UINT64_MAX;
    assert(cast_single_value(input, DTYPE_UINT64).v.u64 == UINT64_MAX);
    assert(cast_single_value(input, DTYPE_INT64).dtype == DTYPE_UNKNOWN);
    puts("scalar conversion checks passed");
    return 0;
}
