#ifndef PHPTENSORS_OPERATOR_OVERLOAD_CE_H
#define PHPTENSORS_OPERATOR_OVERLOAD_CE_H

#include "php.h"

ZEND_FUNCTION(php_tensors_operator_unsupported);

#include "operator_overload_arginfo.h"

void register_operator_overload_class(void);

#endif