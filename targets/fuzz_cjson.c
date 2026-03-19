/*
 * fuzz_cjson.c — AFL++ harness for cJSON
 *
 * Build:
 *   afl-clang-fast -std=c11 -fsanitize=address,undefined -g \
 *       -I targets/src/cJSON \
 *       targets/fuzz_cjson.c targets/src/cJSON/cJSON.c \
 *       -o targets/bin/fuzz_cjson
 */

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "cJSON.h"

__AFL_FUZZ_INIT();

int main(void) {
#ifdef __AFL_HAVE_MANUAL_CONTROL
    __AFL_INIT();
#endif
    unsigned char *buf = __AFL_FUZZ_TESTCASE_BUF;

    while (__AFL_LOOP(10000)) {
        size_t len = (size_t)__AFL_FUZZ_TESTCASE_LEN;
        if (len == 0 || len > (1 << 20)) continue;

        char *input = (char *)malloc(len + 1);
        if (!input) continue;
        memcpy(input, buf, len);
        input[len] = '\0';

        cJSON *json = cJSON_ParseWithLength(input, len);
        if (json) {
            char *out = cJSON_PrintUnformatted(json);
            if (out) {
                cJSON *json2 = cJSON_Parse(out);
                if (json2) cJSON_Delete(json2);
                cJSON_free(out);
            }
            cJSON_Delete(json);
        }
        free(input);
    }
    return 0;
}
