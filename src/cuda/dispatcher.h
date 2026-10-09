#ifndef DISPATCHER_H
#define DISPATCHER_H

#define DISPATCH_DTYPE(DTYPE, ...) \
    switch (DTYPE)                 \
    {                              \
    case DTYPE_FLOAT32:            \
    {                              \
        typedef float scalar_t;    \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_FLOAT64:            \
    {                              \
        typedef double scalar_t;   \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_INT8:               \
    {                              \
        typedef int8_t scalar_t;   \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_INT16:              \
    {                              \
        typedef int16_t scalar_t;  \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_INT32:              \
    {                              \
        typedef int32_t scalar_t;  \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_INT64:              \
    {                              \
        typedef int64_t scalar_t;  \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_UINT8:              \
    {                              \
        typedef uint8_t scalar_t;  \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_UINT16:             \
    {                              \
        typedef uint16_t scalar_t; \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_UINT32:             \
    {                              \
        typedef uint32_t scalar_t; \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_UINT64:             \
    {                              \
        typedef uint64_t scalar_t; \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_BOOL:               \
    {                              \
        typedef bool scalar_t;     \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    default:                       \
        return;                    \
    }

#define DISPATCH_OP(OP_TYPE, ...)          \
    switch (OP_TYPE)                       \
    {                                      \
    case OP_ADD:                           \
    {                                      \
        typedef AddOpT<scalar_t> bin_op_t; \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_SUB:                           \
    {                                      \
        typedef SubOpT<scalar_t> bin_op_t; \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_MUL:                           \
    {                                      \
        typedef MulOpT<scalar_t> bin_op_t; \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_DIV:                           \
    {                                      \
        typedef DivOpT<scalar_t> bin_op_t; \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_POW:                           \
    {                                      \
        typedef PowOpT<scalar_t> bin_op_t; \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_MAXIMUM:                       \
    {                                      \
        typedef MaxOpT<scalar_t> bin_op_t; \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_MINIMUM:                       \
    {                                      \
        typedef MinOpT<scalar_t> bin_op_t; \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_GT:                            \
    {                                      \
        typedef GTOpT<scalar_t> bin_op_t;  \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_LT:                            \
    {                                      \
        typedef LTOpT<scalar_t> bin_op_t;  \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_EQ:                            \
    {                                      \
        typedef EQOpT<scalar_t> bin_op_t;  \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_NE:                            \
    {                                      \
        typedef NEOpT<scalar_t> bin_op_t;  \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_GE:                            \
    {                                      \
        typedef GEOpT<scalar_t> bin_op_t;  \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_LE:                            \
    {                                      \
        typedef LEOpT<scalar_t> bin_op_t;  \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    default:                               \
        return;                            \
    }

#define DISPATCH_UNARY_OP(OP_TYPE, ...)      \
    switch (OP_TYPE)                         \
    {                                        \
    case OP_EXP:                             \
    {                                        \
        typedef ExpOpT<scalar_t> bin_op_t;   \
        __VA_ARGS__;                         \
        break;                               \
    }                                        \
    case OP_SQRT:                            \
    {                                        \
        typedef SqrtOpT<scalar_t> bin_op_t;  \
        __VA_ARGS__;                         \
        break;                               \
    }                                        \
    case OP_LOG:                             \
    {                                        \
        typedef LogOpT<scalar_t> bin_op_t;   \
        __VA_ARGS__;                         \
        break;                               \
    }                                        \
    case OP_SIN:                             \
    {                                        \
        typedef SinOpT<scalar_t> bin_op_t;   \
        __VA_ARGS__;                         \
        break;                               \
    }                                        \
    case OP_COS:                             \
    {                                        \
        typedef CosOpT<scalar_t> bin_op_t;   \
        __VA_ARGS__;                         \
        break;                               \
    }                                        \
    case OP_TAN:                             \
    {                                        \
        typedef TanOpT<scalar_t> bin_op_t;   \
        __VA_ARGS__;                         \
        break;                               \
    }                                        \
    case OP_ABS:                             \
    {                                        \
        typedef AbsOpT<scalar_t> bin_op_t;   \
        __VA_ARGS__;                         \
        break;                               \
    }                                        \
    case OP_NEG:                             \
    {                                        \
        typedef NegOpT<scalar_t> bin_op_t;   \
        __VA_ARGS__;                         \
        break;                               \
    }                                        \
    case OP_FLOOR:                           \
    {                                        \
        typedef FloorOpT<scalar_t> bin_op_t; \
        __VA_ARGS__;                         \
        break;                               \
    }                                        \
    case OP_CEIL:                            \
    {                                        \
        typedef CeilOpT<scalar_t> bin_op_t;  \
        __VA_ARGS__;                         \
        break;                               \
    }                                        \
    case OP_ROUND:                           \
    {                                        \
        typedef RoundOpT<scalar_t> bin_op_t; \
        __VA_ARGS__;                         \
        break;                               \
    }                                        \
    default:                                 \
        return;                              \
    }

#define DISPATCH_OP_REDUCTION(OP_TYPE, ...) \
    switch (OP_TYPE)                        \
    {                                       \
    case OP_REDUCE_SUM:                     \
    {                                       \
        typedef AddOpT<scalar_t> bin_op_t;  \
        __VA_ARGS__;                        \
        break;                              \
    }                                       \
    case OP_REDUCE_MAX:                     \
    {                                       \
        typedef MaxOpT<scalar_t> bin_op_t;  \
        __VA_ARGS__;                        \
        break;                              \
    }                                       \
    case OP_REDUCE_MIN:                     \
    {                                       \
        typedef MinOpT<scalar_t> bin_op_t;  \
        __VA_ARGS__;                        \
        break;                              \
    }                                       \
    case OP_REDUCE_PROD:                    \
    {                                       \
        typedef MulOpT<scalar_t> bin_op_t;  \
        __VA_ARGS__;                        \
        break;                              \
    }                                       \
    default:                                \
        return;                             \
    }

#define DISPATCH_OP_ARG_REDUCTION(OP_TYPE, ...) \
    switch (OP_TYPE)                            \
    {                                           \
    case OP_ARG_MAX:                            \
    {                                           \
        typedef ArgMaxOpT<scalar_t> bin_op_t;   \
        __VA_ARGS__;                            \
        break;                                  \
    }                                           \
    case OP_ARG_MIN:                            \
    {                                           \
        typedef ArgMinOpT<scalar_t> bin_op_t;   \
        __VA_ARGS__;                            \
        break;                                  \
    }                                           \
    default:                                    \
        return;                                 \
    }

#endif