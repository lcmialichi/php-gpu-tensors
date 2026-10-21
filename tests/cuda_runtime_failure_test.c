#include "php.h"
#include "zend_exceptions.h"
#include "memory_pool.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#undef NDEBUG
#include <assert.h>
#undef snprintf
#undef vsnprintf

static zend_executor_globals test_executor;
#undef EG
#define EG(field) test_executor.field

static int failed, claims, thrown, logged;
static char message[512];
static zend_object test_exception;
static zend_class_entry test_exception_class;
zend_class_entry *cuda_runtime_exception_ce = &test_exception_class;

int tensor_mem_failed(void) { return failed; }
void tensor_mem_mark_failed(void) { failed = 1; }
int tensor_mem_claim_owner(void) { claims++; return CUDA_OWNER_CURRENT; }

zend_object *zend_throw_exception_ex(zend_class_entry *ce, zend_long code, const char *format, ...)
{
    assert(ce == cuda_runtime_exception_ce && code == 0 && !EG(exception));
    va_list args;
    va_start(args, format);
    vsnprintf(message, sizeof(message), format, args);
    va_end(args);
    thrown++;
    return EG(exception) = &test_exception;
}

static void test_log_error(const char *text)
{
    snprintf(message, sizeof(message), "%s", text);
    logged++;
}

#undef php_log_err
#define php_log_err(text) test_log_error(text)
#include "../src/tensor.c"

int main(void)
{
    assert(cuda_require_runtime_owner() && claims == 1);
    cuda_fail_runtime("mock kernel synchronization", 719);
    assert(failed && thrown == 1 && logged == 0 && EG(exception) == &test_exception);
    assert(strstr(message, "status 719") && strstr(message, "restart the PHP process"));

    EG(exception) = NULL;
    assert(!cuda_require_runtime_owner());
    assert(claims == 1 && thrown == 2 && strstr(message, "restart the PHP process"));

    zend_object original_exception;
    EG(exception) = &original_exception;
    cuda_fail_runtime("mock cleanup synchronization", 201);
    assert(failed && thrown == 2 && logged == 1 && EG(exception) == &original_exception);
    assert(strstr(message, "status 201") && strstr(message, "restart the PHP process"));
    EG(exception) = NULL;
    puts("CUDA runtime failure blocks reuse, requests restart and preserves an existing exception");
    return 0;
}
