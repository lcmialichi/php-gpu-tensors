#include "kernel_ce.h"
#include "kernel_arginfo.h"
#include "php.h"

zend_class_entry *kernel_ce;
static zend_object_handlers kernel_handlers;

static zend_object *kernel_create_object(zend_class_entry *ce)
{
    kernel_object *obj = emalloc(sizeof(kernel_object));
    zend_object_std_init(&obj->std, ce);
    obj->std.handlers = &kernel_handlers;
    return &obj->std;
}

static void kernel_free_object(zend_object *object)
{
    kernel_object *obj = (kernel_object *)object;
    if (obj->ptx_code)
    {
        efree(obj->ptx_code);
    }

    if (obj->async_operations)
    {
        zend_hash_destroy(obj->async_operations);
        efree(obj->async_operations);
    }

    if (obj->config_cache)
    {
        zend_hash_destroy(obj->config_cache);
        efree(obj->config_cache);
    }

    if (obj->stream_pool)
    {
        efree(obj->stream_pool);
    }

    zend_object_std_dtor(&obj->std);

    efree(obj);
}

int kernel_init(void)
{
    zend_class_entry ce;

    INIT_CLASS_ENTRY(ce, KERNEL_CLASS_NAME, kernel_methods);
    kernel_ce = zend_register_internal_class(&ce);
    kernel_ce->create_object = kernel_create_object;
    kernel_ce->ce_flags |= ZEND_ACC_FINAL;

    memcpy(&kernel_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    kernel_handlers.offset = XtOffsetOf(kernel_object, std);
    kernel_handlers.free_obj = kernel_free_object;

    return 1;
}