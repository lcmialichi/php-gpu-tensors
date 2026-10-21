#include "php.h"
#include "operator_overload_ce.h"
#include "backend_contract.h"
#include "operator_result.h"
#include "zend_exceptions.h"
#include "zend_interfaces.h"
#include <string.h>

static zend_object_handlers operator_overload_handlers;
static zend_class_entry *operator_overload_ce;

ZEND_FUNCTION(php_tensors_operator_unsupported) {
    zval *left, *right;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(left)
        Z_PARAM_ZVAL(right)
    ZEND_PARSE_PARAMETERS_END();
    zend_throw_exception_ex(php_tensors_runtime_exception_ce, 0,
        "%s::%s() is not implemented; override this hook to support the operator",
        ZSTR_VAL(Z_OBJCE_P(ZEND_THIS)->name),
        ZSTR_VAL(execute_data->func->common.function_name));
    RETURN_THROWS();
}

static zend_result operator_overload_do_operation(zend_uchar opcode, zval *result,
                                                  zval *op1, zval *op2) {
    const char *method = NULL;

    switch (opcode) {
        case ZEND_ADD: method = "__add"; break;
        case ZEND_SUB: method = "__sub"; break;
        case ZEND_MUL: method = "__mul"; break;
        case ZEND_DIV: method = "__div"; break;
        case ZEND_POW: method = "__pow"; break;
        case ZEND_MOD: method = "__mod"; break;

        default: return FAILURE;
    }

    zval *obj_ptr = (Z_TYPE_P(op1) == IS_OBJECT &&
                     instanceof_function(Z_OBJCE_P(op1), operator_overload_ce))
        ? op1
        : (op2 && Z_TYPE_P(op2) == IS_OBJECT &&
           instanceof_function(Z_OBJCE_P(op2), operator_overload_ce) ? op2 : NULL);
    if (!obj_ptr) return FAILURE;

    zval value;
    ZVAL_UNDEF(&value);
    zend_call_method(Z_OBJ_P(obj_ptr), Z_OBJCE_P(obj_ptr), NULL, method,
                     strlen(method), &value, 2, op1, op2);
    zend_result status = Z_TYPE(value) != IS_UNDEF ? SUCCESS : FAILURE;
    return php_tensors_finish_operation(status, result, &value, op1, op2);
}

static zend_object *operator_overload_create_object(zend_class_entry *ce) {
    zend_object *obj = zend_objects_new(ce);
    obj->handlers = &operator_overload_handlers;
    object_properties_init(obj, ce);
    return obj;
}

static zend_object *operator_overload_clone_object(zend_object *object) {
    zend_object *clone = operator_overload_create_object(object->ce);
    zend_objects_clone_members(clone, object);
    return clone;
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
    operator_overload_handlers.clone_obj = operator_overload_clone_object;
}