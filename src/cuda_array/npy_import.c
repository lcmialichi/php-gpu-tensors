#include "tensor_import.h"
#include "cuda_exceptions.h"
#include <ctype.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>

typedef struct
{
    const char *cursor;
    const char *end;
} npy_reader;

static void npy_skip_spaces(npy_reader *reader)
{
    while (reader->cursor < reader->end && isspace((unsigned char)*reader->cursor))
        reader->cursor++;
}

static int npy_accept(npy_reader *reader, char expected)
{
    npy_skip_spaces(reader);
    if (reader->cursor >= reader->end || *reader->cursor != expected)
        return 0;
    reader->cursor++;
    return 1;
}

static int npy_quoted(npy_reader *reader, char *value, size_t capacity)
{
    npy_skip_spaces(reader);
    if (reader->cursor >= reader->end || (*reader->cursor != '\'' && *reader->cursor != '"'))
        return 0;
    char quote = *reader->cursor++;
    size_t length = 0;
    while (reader->cursor < reader->end && *reader->cursor != quote)
    {
        char ch = *reader->cursor++;
        if (ch == '\\' || length + 1 >= capacity)
            return 0;
        value[length++] = ch;
    }
    if (reader->cursor == reader->end)
        return 0;
    reader->cursor++;
    value[length] = '\0';
    return 1;
}

static int npy_word(npy_reader *reader, const char *word)
{
    npy_skip_spaces(reader);
    size_t length = strlen(word);
    if ((size_t)(reader->end - reader->cursor) < length ||
        memcmp(reader->cursor, word, length) != 0)
        return 0;
    reader->cursor += length;
    return 1;
}

static int npy_shape(npy_reader *reader, int shape[MAX_DIMS], int *ndims, size_t *elements)
{
    if (!npy_accept(reader, '('))
        return 0;
    *ndims = 0;
    *elements = 1;
    while (1)
    {
        npy_skip_spaces(reader);
        if (npy_accept(reader, ')'))
            return *ndims > 0;
        if (*ndims == MAX_DIMS || reader->cursor == reader->end || !isdigit((unsigned char)*reader->cursor))
            return 0;
        unsigned int dimension = 0;
        while (reader->cursor < reader->end && isdigit((unsigned char)*reader->cursor))
        {
            unsigned int digit = (unsigned int)(*reader->cursor++ - '0');
            if (dimension > (INT_MAX - digit) / 10)
                return 0;
            dimension = dimension * 10 + digit;
        }
        if (!dimension || *elements > SIZE_MAX / dimension)
            return 0;
        shape[(*ndims)++] = (int)dimension;
        *elements *= dimension;
        if (npy_accept(reader, ','))
            continue;
        if (npy_accept(reader, ')'))
            return *ndims > 1;
        return 0;
    }
}

static dtype_t npy_dtype(const char *descr)
{
    if (strlen(descr) != 3 || (descr[0] != '<' && descr[0] != '=' && descr[0] != '|'))
        return DTYPE_UNKNOWN;
    if (descr[0] == '=' && *(const unsigned char *)&(const uint16_t){1} != 1)
        return DTYPE_UNKNOWN;
    if (descr[0] == '|' && descr[2] != '1')
        return DTYPE_UNKNOWN;

    if (strcmp(descr + 1, "f4") == 0) return DTYPE_FLOAT32;
    if (strcmp(descr + 1, "f8") == 0) return DTYPE_FLOAT64;
    if (strcmp(descr + 1, "f2") == 0) return DTYPE_FLOAT16;
    if (strcmp(descr + 1, "i1") == 0) return DTYPE_INT8;
    if (strcmp(descr + 1, "i2") == 0) return DTYPE_INT16;
    if (strcmp(descr + 1, "i4") == 0) return DTYPE_INT32;
    if (strcmp(descr + 1, "i8") == 0) return DTYPE_INT64;
    if (strcmp(descr + 1, "u1") == 0) return DTYPE_UINT8;
    if (strcmp(descr + 1, "u2") == 0) return DTYPE_UINT16;
    if (strcmp(descr + 1, "u4") == 0) return DTYPE_UINT32;
    if (strcmp(descr + 1, "u8") == 0) return DTYPE_UINT64;
    if (strcmp(descr + 1, "b1") == 0) return DTYPE_BOOL;
    return DTYPE_UNKNOWN;
}

static int npy_parse_header(const char *header, size_t length,
                            int shape[MAX_DIMS], int *ndims, size_t *elements, dtype_t *dtype)
{
    npy_reader reader = {header, header + length};
    int found_descr = 0, found_shape = 0, found_order = 0;
    if (!npy_accept(&reader, '{')) return 0;

    while (1)
    {
        npy_skip_spaces(&reader);
        if (npy_accept(&reader, '}')) break;
        char key[32];
        if (!npy_quoted(&reader, key, sizeof(key)) || !npy_accept(&reader, ':')) return 0;

        if (strcmp(key, "descr") == 0 && !found_descr)
        {
            char descr[8];
            if (!npy_quoted(&reader, descr, sizeof(descr))) return 0;
            *dtype = npy_dtype(descr);
            if (*dtype == DTYPE_UNKNOWN) return 0;
            found_descr = 1;
        }
        else if (strcmp(key, "fortran_order") == 0 && !found_order)
        {
            if (!npy_word(&reader, "False")) return 0;
            found_order = 1;
        }
        else if (strcmp(key, "shape") == 0 && !found_shape)
        {
            if (!npy_shape(&reader, shape, ndims, elements)) return 0;
            found_shape = 1;
        }
        else return 0;

        if (npy_accept(&reader, ',')) continue;
        if (npy_accept(&reader, '}')) break;
        return 0;
    }

    npy_skip_spaces(&reader);
    return found_descr && found_order && found_shape && reader.cursor == reader.end;
}

static int npy_read_exact(php_stream *stream, void *buffer, size_t bytes)
{
    size_t offset = 0;
    while (offset < bytes)
    {
        size_t received = php_stream_read(stream, (char *)buffer + offset, bytes - offset);
        if (!received) return 0;
        offset += received;
    }
    return 1;
}

tensor_t *tensor_import_npy(zend_string *path)
{
    php_stream *stream = php_stream_open_wrapper(ZSTR_VAL(path), "rb", 0, NULL);
    if (!stream)
    {
        CUDA_THROW_RUNTIME("Cannot open NumPy file: %s", ZSTR_VAL(path));
        return NULL;
    }

    unsigned char prefix[12];
    size_t prefix_size = 10;
    int valid = npy_read_exact(stream, prefix, 8) &&
                memcmp(prefix, "\x93NUMPY", 6) == 0 &&
                prefix[6] >= 1 && prefix[6] <= 3 && prefix[7] == 0;
    if (valid && prefix[6] >= 2) prefix_size = 12;
    if (valid) valid = npy_read_exact(stream, prefix + 8, prefix_size - 8);

    size_t header_length = 0;
    if (valid)
    {
        header_length = (size_t)prefix[8] | ((size_t)prefix[9] << 8);
        if (prefix_size == 12)
            header_length |= ((size_t)prefix[10] << 16) | ((size_t)prefix[11] << 24);
        valid = header_length > 0 && header_length <= 1024 * 1024;
    }

    char *header = NULL;
    if (valid)
    {
        header = emalloc(header_length);
        valid = npy_read_exact(stream, header, header_length);
    }

    int shape[MAX_DIMS];
    int ndims = 0;
    size_t elements = 0;
    dtype_t dtype = DTYPE_UNKNOWN;
    if (valid) valid = npy_parse_header(header, header_length, shape, &ndims, &elements, &dtype);
    if (header) efree(header);

    tensor_t *tensor = NULL;
    if (valid && elements <= SIZE_MAX / dtype_size(dtype))
        tensor = tensor_import_stream(stream, shape, ndims, dtype, elements * dtype_size(dtype));
    else
        CUDA_THROW_INVALID("Unsupported or malformed NumPy array (requires C-order, little-endian, supported dtype)");

    php_stream_close(stream);
    return tensor;
}