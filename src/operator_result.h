#ifndef PHPTENSORS_OPERATOR_RESULT_H
#define PHPTENSORS_OPERATOR_RESULT_H

#include "php.h"

static zend_always_inline zend_result php_tensors_finish_operation(
    zend_result status, zval *result, zval *value, zval *op1, zval *op2)
{
    if (status != SUCCESS || EG(exception)) {
        zval_ptr_dtor(value);
        ZVAL_UNDEF(value);
        if (EG(exception)) {
            if (result != op1 && result != op2)
                ZVAL_UNDEF(result);
            /* Handled exceptions must not trigger PHP's increment/cast fallback. */
            return SUCCESS;
        }
        return FAILURE;
    }
    if (Z_ISREF_P(value)) {
        zval dereferenced;
        ZVAL_COPY_DEREF(&dereferenced, value);
        zval_ptr_dtor(value);
        ZVAL_COPY_VALUE(value, &dereferenced);
    }

    zval previous;
    ZVAL_UNDEF(&previous);
    if (result == op1 || result == op2)
        ZVAL_COPY_VALUE(&previous, result);
    ZVAL_COPY_VALUE(result, value);
    ZVAL_UNDEF(value);
    /* Destructors must observe the new binding, not a dying operand. */
    zval_ptr_dtor(&previous);
    return SUCCESS;
}

#endif
