/*
 * fuzz_yyjson.c — AFL++ harness for yyjson
 *
 * Build:
 *   afl-clang-fast -std=c11 -fsanitize=address,undefined -g \
 *       -I targets/src/yyjson/src \
 *       targets/fuzz_yyjson.c targets/src/yyjson/src/yyjson.c \
 *       -o targets/bin/fuzz_yyjson
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>
#include "yyjson.h"

__AFL_FUZZ_INIT();

static void traverse_val(yyjson_val *val, int depth) {
    if (!val || depth > 128) return;
    yyjson_type type = yyjson_get_type(val);
    switch (type) {
        case YYJSON_TYPE_OBJ: {
            yyjson_obj_iter iter;
            yyjson_obj_iter_init(val, &iter);
            /* используем safe API */
            size_t idx = 0, max = yyjson_obj_size(val);
            yyjson_val *key = yyjson_obj_iter_next(&iter);
            while (key && idx++ < max) {
                yyjson_val *v = yyjson_obj_iter_get_val(key);
                traverse_val(v, depth + 1);
                key = yyjson_obj_iter_next(&iter);
            }
            break;
        }
        case YYJSON_TYPE_ARR: {
            size_t idx = 0, max = yyjson_arr_size(val);
            yyjson_arr_iter iter;
            yyjson_arr_iter_init(val, &iter);
            yyjson_val *v;
            while ((v = yyjson_arr_iter_next(&iter)) && idx++ < max)
                traverse_val(v, depth + 1);
            break;
        }
        case YYJSON_TYPE_STR:
            (void)yyjson_get_str(val);
            (void)yyjson_get_len(val);
            break;
        case YYJSON_TYPE_NUM:
            (void)yyjson_get_real(val);
            (void)yyjson_get_int(val);
            (void)yyjson_get_uint(val);
            break;
        default:
            break;
    }
}

int main(void) {
#ifdef __AFL_HAVE_MANUAL_CONTROL
    __AFL_INIT();
#endif
    unsigned char *buf = __AFL_FUZZ_TESTCASE_BUF;

    while (__AFL_LOOP(10000)) {
        size_t len = (size_t)__AFL_FUZZ_TESTCASE_LEN;
        if (len == 0 || len > (1 << 20)) continue;

        /* Path 1: strict parse */
        yyjson_read_flag flags = YYJSON_READ_NOFLAG;
        yyjson_doc *doc = yyjson_read((char *)buf, len, flags);
        if (doc) {
            traverse_val(yyjson_doc_get_root(doc), 0);

            /* Path 2: write round-trip */
            size_t out_len;
            char *out = yyjson_write(doc, YYJSON_WRITE_NOFLAG, &out_len);
            if (out) {
                yyjson_doc *doc2 = yyjson_read(out, out_len, flags);
                if (doc2) yyjson_doc_free(doc2);
                free(out);
            }
            yyjson_doc_free(doc);
        }

        /* Path 3: insitu parse (mutable, разные флаги) */
        char *copy = (char *)malloc(len + 4);
        if (copy) {
            memcpy(copy, buf, len);
            yyjson_doc *doc3 = yyjson_read_opts(copy, len, YYJSON_READ_INSITU,
                                                NULL, NULL);
            if (doc3) yyjson_doc_free(doc3);
            free(copy);
        }
    }
    return 0;
}
