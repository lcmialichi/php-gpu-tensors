#ifndef OPERATIONS_STRUCTURES_H
#define OPERATIONS_STRUCTURES_H

typedef enum
{
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_POW,
    OP_EXP,
    OP_SQRT,
    OP_LOG,
    OP_SIN,
    OP_COS,
    OP_TAN,
    OP_ABS,
    OP_NEG,
    OP_SELECT,
    OP_CLAMP,
    OP_CEIL,
    OP_FLOOR,
    OP_ROUND,

    OP_GT,
    OP_LT,
    OP_EQ,
    OP_NE,
    OP_GE,
    OP_LE,

    OP_REDUCE_SUM,
    OP_REDUCE_MEAN,
    OP_REDUCE_MAX,
    OP_REDUCE_MIN,
    OP_REDUCE_PROD,

    OP_ARG_MAX,
    OP_ARG_MIN,

    OP_CONCAT,

    OP_RESHAPE,
    OP_TRANSPOSE,
    OP_SLICE,

    OP_MATMUL,

    OP_MAXIMUM,
    OP_MINIMUM,
    OP_BROADCAST,
    OP_REDUCE_ALL,
    OP_REDUCE_ANY
} operation_type_t;

#endif