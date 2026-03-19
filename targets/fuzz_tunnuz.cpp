/*
 * fuzz_tunnuz.cpp — AFL++ harness for tunnuz/json (json++)
 *
 * Build:
 *   afl-clang-fast++ -std=c++11 -fsanitize=address,undefined -g \
 *       -Wno-pessimizing-move -Wno-unneeded-internal-declaration \
 *       -I targets/src/tunnuz_json \
 *       -I targets/src/tunnuz_json/build \
 *       targets/fuzz_tunnuz.cpp \
 *       targets/src/tunnuz_json/build/json.tab.cc \
 *       targets/src/tunnuz_json/build/lex.yy.cc \
 *       targets/src/tunnuz_json/json_st.cc \
 *       -o targets/bin/fuzz_tunnuz
 */

#include <string>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include "json.hh"

using namespace JSON;

__AFL_FUZZ_INIT();

static void traverse(const Value &v, int depth) {
    if (depth > 64) return;
    switch (v.type()) {
        case OBJECT: {
            const Object &obj = static_cast<Object>(v);
            for (auto it = obj.begin(); it != obj.end(); ++it)
                traverse(it->second, depth + 1);
            break;
        }
        case ARRAY: {
            const Array &arr = static_cast<Array>(v);
            for (size_t i = 0; i < arr.size(); i++)
                traverse(arr[i], depth + 1);
            break;
        }
        case STRING: { std::string s = static_cast<std::string>(v); (void)s; break; }
        case INT:    { long long i = static_cast<long long>(v); (void)i; break; }
        case BOOL:   { bool b = static_cast<bool>(v); (void)b; break; }
        case FLOAT:  { std::ostringstream oss; oss << v; break; }
        default: break;
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

        std::string input(reinterpret_cast<char *>(buf), len);
        try {
            Value v = parse_string(input);
            traverse(v, 0);
            std::ostringstream oss;
            oss << v;
            try { Value v2 = parse_string(oss.str()); (void)v2; } catch (...) {}
        } catch (const std::exception &) {
        } catch (...) {}
    }
    return 0;
}
