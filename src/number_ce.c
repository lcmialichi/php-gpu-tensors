#include "php.h"
#include "number_ce.h"
#include "backend_contract.h"
#include "zend_interfaces.h"
#include <string.h>

static zend_object_handlers number_handlers;
zend_class_entry *number_ce;

static zend_result number_do_operation(zend_uchar opcode, zval *result, zval *op1, zval *op2) {
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

    zval *obj_ptr = (Z_TYPE_P(op1) == IS_OBJECT && instanceof_function(Z_OBJCE_P(op1), number_ce)) ? op1 : op2;
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

static zend_object* number_create_object(zend_class_entry *ce) {
    zend_object *obj = zend_objects_new(ce);
    obj->handlers = &number_handlers;
    object_properties_init(obj, ce);
    return obj;
}

void register_number_class() {
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, "PhpTensors\\Number", number_methods);
    number_ce = zend_register_internal_class(&ce);
    number_ce->ce_flags |= ZEND_ACC_ABSTRACT;
    number_ce->create_object = number_create_object;
    zend_class_implements(number_ce, 1, php_tensors_numbers_ce);

    memcpy(&number_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    number_handlers.do_operation = number_do_operation;
}