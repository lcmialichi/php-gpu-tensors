#include "php.h"
#include "operator_overload_ce.h"
#include "backend_contract.h"
#include "zend_interfaces.h"
#include <string.h>

static zend_object_handlers operator_overload_handlers;
static zend_class_entry *operator_overload_ce;

static zend_result operator_overload_do_operation(zend_uchar opcode, zval *result,
                                                  zval *op1, zval *op2) {
    const char *method = NULL;
    uint32_t param_count = 2;

    switch (opcode) {
        case ZEND_ADD: method = "__add"; break;
        case ZEND_SUB: method = "__sub"; break;
        case ZEND_MUL: method = "__mul"; break;
        case ZEND_DIV: method = "__div"; break;
        case ZEND_POW: method = "__pow"; break;
        case ZEND_MOD: method = "__mod"; break;

        case ZEND_PRE_INC:
        case ZEND_POST_INC:
            method = "__inc";
            param_count = 0;
            break;
        case ZEND_PRE_DEC:
        case ZEND_POST_DEC:
            method = "__dec";
            param_count = 0;
            break;

        default: return FAILURE;
    }

    zval *obj_ptr = (Z_TYPE_P(op1) == IS_OBJECT &&
                     instanceof_function(Z_OBJCE_P(op1), operator_overload_ce))
        ? op1
        : op2;
    if (!obj_ptr) return FAILURE;

    if (param_count == 2) {
        zend_call_method(Z_OBJ_P(obj_ptr), Z_OBJCE_P(obj_ptr), NULL, method,
                         strlen(method), result, 2, op1, op2);
    } else {
        zend_call_method(Z_OBJ_P(obj_ptr), Z_OBJCE_P(obj_ptr), NULL, method,
                         strlen(method), result, 0, NULL, NULL);
    }

    return (Z_TYPE_P(result) != IS_UNDEF) ? SUCCESS : FAILURE;
}

static zend_object *operator_overload_create_object(zend_class_entry *ce) {
    zend_object *obj = zend_objects_new(ce);
    obj->handlers = &operator_overload_handlers;
    object_properties_init(obj, ce);
    return obj;
}

void register_operator_overload_class(void) {
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, "PhpTensors\\OperatorOverload", operator_overload_methods);
    operator_overload_ce = zend_register_internal_class(&ce);
    operator_overload_ce->ce_flags |= ZEND_ACC_EXPLICIT_ABSTRACT_CLASS;
    operator_overload_ce->create_object = operator_overload_create_object;
    zend_class_implements(operator_overload_ce, 1,
                          php_tensors_operator_overloadable_ce);

    memcpy(&operator_overload_handlers, zend_get_std_object_handlers(),
           sizeof(zend_object_handlers));
    operator_overload_handlers.do_operation = operator_overload_do_operation;
}