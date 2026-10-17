#include "php.h"
#include "backend_contract.h"
#include "zend_exceptions.h"

ZEND_FUNCTION(php_tensors_host_array_unsupported)
{
    zend_throw_exception_ex(
        php_tensors_runtime_exception_ce,
        0,
        "PhpTensors\\HostArray::%s() is not implemented yet",
        ZSTR_VAL(execute_data->func->common.function_name));
    RETURN_THROWS();
}
