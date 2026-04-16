#include "moe.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FLOAT_ATOL 2e-5f
#define FLOAT_RTOL 1e-5f

typedef struct FloatArray {
    float *data;
    int length;
} FloatArray;

typedef struct IntArray {
    int *data;
    int length;
} IntArray;

static char *read_text_file(const char *path) {
    FILE *file = fopen(path, "rb");
    char *buffer;
    long size;

    if (file == NULL) {
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    size = ftell(file);
    if (size < 0) {
        fclose(file);
        return NULL;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }

    buffer = (char *)malloc((size_t)size + 1U);
    if (buffer == NULL) {
        fclose(file);
        return NULL;
    }
    if (fread(buffer, 1, (size_t)size, file) != (size_t)size) {
        free(buffer);
        fclose(file);
        return NULL;
    }
    buffer[size] = '\0';
    fclose(file);
    return buffer;
}

static char *find_after_key(char *start, const char *key) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    return strstr(start, pattern);
}

static int parse_int_after_key(char *start, const char *key, int *value) {
    char *cursor = find_after_key(start, key);
    if (cursor == NULL) {
        return 0;
    }
    cursor = strchr(cursor, ':');
    if (cursor == NULL) {
        return 0;
    }
    cursor += 1;
    while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' || *cursor == '\t') {
        cursor += 1;
    }
    *value = (int)strtol(cursor, NULL, 10);
    return 1;
}

static int parse_bool_after_key(char *start, const char *key, int *value) {
    char *cursor = find_after_key(start, key);
    if (cursor == NULL) {
        return 0;
    }
    cursor = strchr(cursor, ':');
    if (cursor == NULL) {
        return 0;
    }
    cursor += 1;
    while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' || *cursor == '\t') {
        cursor += 1;
    }
    if (strncmp(cursor, "true", 4) == 0) {
        *value = 1;
        return 1;
    }
    if (strncmp(cursor, "false", 5) == 0) {
        *value = 0;
        return 1;
    }
    return 0;
}

static int parse_float_after_key(char *start, const char *key, float *value) {
    char *cursor = find_after_key(start, key);
    if (cursor == NULL) {
        return 0;
    }
    cursor = strchr(cursor, ':');
    if (cursor == NULL) {
        return 0;
    }
    cursor += 1;
    while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' || *cursor == '\t') {
        cursor += 1;
    }
    *value = strtof(cursor, NULL);
    return 1;
}

static FloatArray parse_float_array_after_key(char *start, const char *key) {
    FloatArray result;
    char *cursor = find_after_key(start, key);
    int count = 0;
    int index = 0;
    char *scan;

    result.data = NULL;
    result.length = 0;
    if (cursor == NULL) {
        return result;
    }
    cursor = strchr(cursor, '[');
    if (cursor == NULL) {
        return result;
    }
    cursor += 1;
    scan = cursor;
    while (*scan != '\0' && *scan != ']') {
        char *end_ptr;
        while (*scan != '\0' && (isspace((unsigned char)*scan) || *scan == ',')) {
            scan += 1;
        }
        if (*scan == ']') {
            break;
        }
        strtof(scan, &end_ptr);
        if (end_ptr == scan) {
            break;
        }
        count += 1;
        scan = end_ptr;
    }

    result.data = (float *)malloc((size_t)count * sizeof(float));
    result.length = count;
    scan = cursor;
    while (*scan != '\0' && *scan != ']' && index < count) {
        char *end_ptr;
        while (*scan != '\0' && (isspace((unsigned char)*scan) || *scan == ',')) {
            scan += 1;
        }
        if (*scan == ']') {
            break;
        }
        result.data[index] = strtof(scan, &end_ptr);
        if (end_ptr == scan) {
            break;
        }
        index += 1;
        scan = end_ptr;
    }
    return result;
}

static IntArray parse_int_array_after_key(char *start, const char *key) {
    IntArray result;
    char *cursor = find_after_key(start, key);
    int count = 0;
    int index = 0;
    char *scan;

    result.data = NULL;
    result.length = 0;
    if (cursor == NULL) {
        return result;
    }
    cursor = strchr(cursor, '[');
    if (cursor == NULL) {
        return result;
    }
    cursor += 1;
    scan = cursor;
    while (*scan != '\0' && *scan != ']') {
        char *end_ptr;
        while (*scan != '\0' && (isspace((unsigned char)*scan) || *scan == ',')) {
            scan += 1;
        }
        if (*scan == ']') {
            break;
        }
        strtol(scan, &end_ptr, 10);
        if (end_ptr == scan) {
            break;
        }
        count += 1;
        scan = end_ptr;
    }

    result.data = (int *)malloc((size_t)count * sizeof(int));
    result.length = count;
    scan = cursor;
    while (*scan != '\0' && *scan != ']' && index < count) {
        char *end_ptr;
        while (*scan != '\0' && (isspace((unsigned char)*scan) || *scan == ',')) {
            scan += 1;
        }
        if (*scan == ']') {
            break;
        }
        result.data[index] = (int)strtol(scan, &end_ptr, 10);
        if (end_ptr == scan) {
            break;
        }
        index += 1;
        scan = end_ptr;
    }
    return result;
}

static void free_float_array(FloatArray array) {
    free(array.data);
}

static void free_int_array(IntArray array) {
    free(array.data);
}

static int compare_float_arrays(const float *actual, const float *expected, int length, const char *label) {
    int idx;
    for (idx = 0; idx < length; ++idx) {
        float diff = fabsf(actual[idx] - expected[idx]);
        float limit = FLOAT_ATOL + FLOAT_RTOL * fabsf(expected[idx]);
        if (diff > limit) {
            printf("FAIL %s mismatch at %d actual=%0.8f expected=%0.8f diff=%0.8f\n", label, idx, actual[idx], expected[idx], diff);
            return 0;
        }
    }
    return 1;
}

static int compare_int_arrays(const int *actual, const int *expected, int length, const char *label) {
    int idx;
    for (idx = 0; idx < length; ++idx) {
        if (actual[idx] != expected[idx]) {
            printf("FAIL %s mismatch at %d actual=%d expected=%d\n", label, idx, actual[idx], expected[idx]);
            return 0;
        }
    }
    return 1;
}

static int run_mlp_cases(const char *path) {
    char *json = read_text_file(path);
    char *cursor;
    int hidden_size = 0;
    int intermediate_size = 0;
    int cases_passed = 0;
    int case_count = 0;

    if (json == NULL) {
        printf("FAIL unable to read %s\n", path);
        return 1;
    }

    if (!parse_int_after_key(json, "hidden_size", &hidden_size) ||
        !parse_int_after_key(json, "intermediate_size", &intermediate_size)) {
        printf("FAIL unable to parse meta config from %s\n", path);
        free(json);
        return 1;
    }

    cursor = json;
    while ((cursor = find_after_key(cursor, "name")) != NULL) {
        char *next_case = find_after_key(cursor + 1, "name");
        char saved_char = '\0';
        FloatArray hidden_states;
        IntArray hidden_states_shape;
        FloatArray gate_proj_weight;
        FloatArray up_proj_weight;
        FloatArray down_proj_weight;
        FloatArray expected;
        float *actual;
        int num_tokens;
        int ok;

        if (next_case != NULL) {
            saved_char = *next_case;
            *next_case = '\0';
        }
        hidden_states = parse_float_array_after_key(cursor, "hidden_states");
        hidden_states_shape = parse_int_array_after_key(cursor, "hidden_states_shape");
        gate_proj_weight = parse_float_array_after_key(cursor, "gate_proj_weight");
        up_proj_weight = parse_float_array_after_key(cursor, "up_proj_weight");
        down_proj_weight = parse_float_array_after_key(cursor, "down_proj_weight");
        expected = parse_float_array_after_key(cursor, "mlp_out");
        if (next_case != NULL) {
            *next_case = saved_char;
        }

        if (hidden_states_shape.length != 3) {
            printf("FAIL invalid hidden_states_shape in case %d\n", case_count);
            free(json);
            return 1;
        }
        num_tokens = hidden_states_shape.data[0] * hidden_states_shape.data[1];
        actual = (float *)malloc((size_t)expected.length * sizeof(float));
        deepseek_mlp_forward(
            hidden_states.data,
            num_tokens,
            hidden_size,
            intermediate_size,
            gate_proj_weight.data,
            up_proj_weight.data,
            down_proj_weight.data,
            actual
        );

        ok = compare_float_arrays(actual, expected.data, expected.length, "mlp");
        if (!ok) {
            printf("FAIL case_%d\n", case_count);
            free(actual);
            free_float_array(hidden_states);
            free_int_array(hidden_states_shape);
            free_float_array(gate_proj_weight);
            free_float_array(up_proj_weight);
            free_float_array(down_proj_weight);
            free_float_array(expected);
            free(json);
            return 1;
        }

        cases_passed += 1;
        case_count += 1;
        free(actual);
        free_float_array(hidden_states);
        free_int_array(hidden_states_shape);
        free_float_array(gate_proj_weight);
        free_float_array(up_proj_weight);
        free_float_array(down_proj_weight);
        free_float_array(expected);
        cursor = next_case != NULL ? next_case : cursor + 6;
    }

    printf("PASS mlp %d cases\n", cases_passed);
    free(json);
    return 0;
}

static int run_router_cases(const char *path) {
    char *json = read_text_file(path);
    char *cursor;
    int hidden_size = 0;
    int n_routed_experts = 0;
    int cases_passed = 0;
    int case_count = 0;

    if (json == NULL) {
        printf("FAIL unable to read %s\n", path);
        return 1;
    }

    if (!parse_int_after_key(json, "hidden_size", &hidden_size) ||
        !parse_int_after_key(json, "n_routed_experts", &n_routed_experts)) {
        printf("FAIL unable to parse router meta config from %s\n", path);
        free(json);
        return 1;
    }

    cursor = json;
    while ((cursor = find_after_key(cursor, "name")) != NULL) {
        char *next_case = find_after_key(cursor + 1, "name");
        char saved_char = '\0';
        FloatArray hidden_states;
        IntArray hidden_states_shape;
        FloatArray router_weight;
        FloatArray expected;
        float *actual;
        int num_tokens;
        int ok;

        if (next_case != NULL) {
            saved_char = *next_case;
            *next_case = '\0';
        }
        hidden_states = parse_float_array_after_key(cursor, "hidden_states");
        hidden_states_shape = parse_int_array_after_key(cursor, "hidden_states_shape");
        router_weight = parse_float_array_after_key(cursor, "router_weight");
        expected = parse_float_array_after_key(cursor, "router_logits");
        if (next_case != NULL) {
            *next_case = saved_char;
        }

        if (hidden_states_shape.length != 3) {
            printf("FAIL invalid hidden_states_shape in router case %d\n", case_count);
            free(json);
            return 1;
        }

        num_tokens = hidden_states_shape.data[0] * hidden_states_shape.data[1];
        actual = (float *)malloc((size_t)expected.length * sizeof(float));
        deepseek_topk_router_forward(
            hidden_states.data,
            num_tokens,
            hidden_size,
            n_routed_experts,
            router_weight.data,
            actual
        );

        ok = compare_float_arrays(actual, expected.data, expected.length, "router");
        if (!ok) {
            printf("FAIL router case_%d\n", case_count);
            free(actual);
            free_float_array(hidden_states);
            free_int_array(hidden_states_shape);
            free_float_array(router_weight);
            free_float_array(expected);
            free(json);
            return 1;
        }

        cases_passed += 1;
        case_count += 1;
        free(actual);
        free_float_array(hidden_states);
        free_int_array(hidden_states_shape);
        free_float_array(router_weight);
        free_float_array(expected);
        cursor = next_case != NULL ? next_case : cursor + 6;
    }

    printf("PASS router %d cases\n", cases_passed);
    free(json);
    return 0;
}

static int run_route_cases(const char *path) {
    char *json = read_text_file(path);
    char *cursor;
    int n_routed_experts = 0;
    int n_group = 0;
    int topk_group = 0;
    int top_k = 0;
    int norm_topk_prob = 0;
    float routed_scaling_factor = 0.0f;
    int cases_passed = 0;
    int case_count = 0;

    if (json == NULL) {
        printf("FAIL unable to read %s\n", path);
        return 1;
    }

    if (!parse_int_after_key(json, "n_routed_experts", &n_routed_experts) ||
        !parse_int_after_key(json, "n_group", &n_group) ||
        !parse_int_after_key(json, "topk_group", &topk_group) ||
        !parse_int_after_key(json, "num_experts_per_tok", &top_k) ||
        !parse_bool_after_key(json, "norm_topk_prob", &norm_topk_prob) ||
        !parse_float_after_key(json, "routed_scaling_factor", &routed_scaling_factor)) {
        printf("FAIL unable to parse route meta config from %s\n", path);
        free(json);
        return 1;
    }

    cursor = json;
    while ((cursor = find_after_key(cursor, "name")) != NULL) {
        char *next_case = find_after_key(cursor + 1, "name");
        char saved_char = '\0';
        FloatArray router_logits;
        FloatArray e_score_correction_bias;
        IntArray topk_indices_expected;
        FloatArray topk_weights_expected;
        IntArray router_logits_shape;
        IntArray topk_indices_actual;
        FloatArray topk_weights_actual;
        int num_tokens;
        int ok_indices;
        int ok_weights;

        if (next_case != NULL) {
            saved_char = *next_case;
            *next_case = '\0';
        }
        router_logits = parse_float_array_after_key(cursor, "router_logits");
        router_logits_shape = parse_int_array_after_key(cursor, "router_logits_shape");
        e_score_correction_bias = parse_float_array_after_key(cursor, "e_score_correction_bias");
        topk_indices_expected = parse_int_array_after_key(cursor, "topk_indices");
        topk_weights_expected = parse_float_array_after_key(cursor, "topk_weights");
        if (next_case != NULL) {
            *next_case = saved_char;
        }

        if (router_logits_shape.length != 2) {
            printf("FAIL invalid router_logits_shape in route case %d\n", case_count);
            free(json);
            return 1;
        }

        num_tokens = router_logits_shape.data[0];
        topk_indices_actual.data = (int *)malloc((size_t)topk_indices_expected.length * sizeof(int));
        topk_indices_actual.length = topk_indices_expected.length;
        topk_weights_actual.data = (float *)malloc((size_t)topk_weights_expected.length * sizeof(float));
        topk_weights_actual.length = topk_weights_expected.length;

        deepseek_route_tokens_to_experts(
            router_logits.data,
            num_tokens,
            n_routed_experts,
            n_group,
            topk_group,
            top_k,
            norm_topk_prob,
            routed_scaling_factor,
            e_score_correction_bias.data,
            topk_indices_actual.data,
            topk_weights_actual.data
        );

        ok_indices = compare_int_arrays(topk_indices_actual.data, topk_indices_expected.data, topk_indices_expected.length, "route_indices");
        ok_weights = compare_float_arrays(topk_weights_actual.data, topk_weights_expected.data, topk_weights_expected.length, "route_weights");
        if (!ok_indices || !ok_weights) {
            printf("FAIL route case_%d\n", case_count);
            free_float_array(router_logits);
            free_int_array(router_logits_shape);
            free_float_array(e_score_correction_bias);
            free_int_array(topk_indices_expected);
            free_float_array(topk_weights_expected);
            free_int_array(topk_indices_actual);
            free_float_array(topk_weights_actual);
            free(json);
            return 1;
        }

        cases_passed += 1;
        case_count += 1;
        free_float_array(router_logits);
        free_int_array(router_logits_shape);
        free_float_array(e_score_correction_bias);
        free_int_array(topk_indices_expected);
        free_float_array(topk_weights_expected);
        free_int_array(topk_indices_actual);
        free_float_array(topk_weights_actual);
        cursor = next_case != NULL ? next_case : cursor + 6;
    }

    printf("PASS route %d cases\n", cases_passed);
    free(json);
    return 0;
}

static int run_naive_moe_cases(const char *path) {
    char *json = read_text_file(path);
    char *cursor;
    int hidden_size = 0;
    int num_local_experts = 0;
    int intermediate_size = 0;
    int top_k = 0;
    int cases_passed = 0;
    int case_count = 0;

    if (json == NULL) {
        printf("FAIL unable to read %s\n", path);
        return 1;
    }

    if (!parse_int_after_key(json, "hidden_size", &hidden_size) ||
        !parse_int_after_key(json, "num_local_experts", &num_local_experts) ||
        !parse_int_after_key(json, "moe_intermediate_size", &intermediate_size) ||
        !parse_int_after_key(json, "num_experts_per_tok", &top_k)) {
        printf("FAIL unable to parse naive moe meta config from %s\n", path);
        free(json);
        return 1;
    }

    cursor = json;
    while ((cursor = find_after_key(cursor, "name")) != NULL) {
        char *next_case = find_after_key(cursor + 1, "name");
        char saved_char = '\0';
        FloatArray hidden_states;
        IntArray hidden_states_shape;
        IntArray top_k_index;
        FloatArray top_k_weights;
        FloatArray gate_up_proj;
        FloatArray down_proj;
        FloatArray expected;
        float *actual;
        int num_tokens;
        int ok;

        if (next_case != NULL) {
            saved_char = *next_case;
            *next_case = '\0';
        }
        hidden_states = parse_float_array_after_key(cursor, "hidden_states");
        hidden_states_shape = parse_int_array_after_key(cursor, "hidden_states_shape");
        top_k_index = parse_int_array_after_key(cursor, "top_k_index");
        top_k_weights = parse_float_array_after_key(cursor, "top_k_weights");
        gate_up_proj = parse_float_array_after_key(cursor, "gate_up_proj");
        down_proj = parse_float_array_after_key(cursor, "down_proj");
        expected = parse_float_array_after_key(cursor, "routed_out");
        if (next_case != NULL) {
            *next_case = saved_char;
        }

        if (hidden_states_shape.length != 2) {
            printf("FAIL invalid hidden_states_shape in naive moe case %d\n", case_count);
            free(json);
            return 1;
        }

        num_tokens = hidden_states_shape.data[0];
        actual = (float *)malloc((size_t)expected.length * sizeof(float));
        deepseek_naive_moe_forward(
            hidden_states.data,
            num_tokens,
            hidden_size,
            num_local_experts,
            intermediate_size,
            top_k,
            top_k_index.data,
            top_k_weights.data,
            gate_up_proj.data,
            down_proj.data,
            actual
        );

        ok = compare_float_arrays(actual, expected.data, expected.length, "naive_moe");
        if (!ok) {
            printf("FAIL naive_moe case_%d\n", case_count);
            free(actual);
            free_float_array(hidden_states);
            free_int_array(hidden_states_shape);
            free_int_array(top_k_index);
            free_float_array(top_k_weights);
            free_float_array(gate_up_proj);
            free_float_array(down_proj);
            free_float_array(expected);
            free(json);
            return 1;
        }

        cases_passed += 1;
        case_count += 1;
        free(actual);
        free_float_array(hidden_states);
        free_int_array(hidden_states_shape);
        free_int_array(top_k_index);
        free_float_array(top_k_weights);
        free_float_array(gate_up_proj);
        free_float_array(down_proj);
        free_float_array(expected);
        cursor = next_case != NULL ? next_case : cursor + 6;
    }

    printf("PASS naive_moe %d cases\n", cases_passed);
    free(json);
    return 0;
}

static int run_moe_cases(const char *path) {
    char *json = read_text_file(path);
    char *cursor;
    DeepseekMoeConfig config;
    int batch_size = 0;
    int seq_len = 0;
    int cases_passed = 0;
    int case_count = 0;

    if (json == NULL) {
        printf("FAIL unable to read %s\n", path);
        return 1;
    }

    if (!parse_int_after_key(json, "hidden_size", &config.hidden_size) ||
        !parse_int_after_key(json, "intermediate_size", &config.intermediate_size) ||
        !parse_int_after_key(json, "moe_intermediate_size", &config.moe_intermediate_size) ||
        !parse_int_after_key(json, "n_routed_experts", &config.n_routed_experts) ||
        !parse_int_after_key(json, "num_local_experts", &config.num_local_experts) ||
        !parse_int_after_key(json, "n_shared_experts", &config.n_shared_experts) ||
        !parse_int_after_key(json, "n_group", &config.n_group) ||
        !parse_int_after_key(json, "topk_group", &config.topk_group) ||
        !parse_int_after_key(json, "num_experts_per_tok", &config.num_experts_per_tok) ||
        !parse_bool_after_key(json, "norm_topk_prob", &config.norm_topk_prob) ||
        !parse_float_after_key(json, "routed_scaling_factor", &config.routed_scaling_factor) ||
        !parse_int_after_key(json, "batch_size", &batch_size) ||
        !parse_int_after_key(json, "seq_len", &seq_len)) {
        printf("FAIL unable to parse moe meta config from %s\n", path);
        free(json);
        return 1;
    }

    cursor = json;
    while ((cursor = find_after_key(cursor, "name")) != NULL) {
        char *next_case = find_after_key(cursor + 1, "name");
        char saved_char = '\0';
        FloatArray hidden_states;
        FloatArray router_weight;
        FloatArray e_score_correction_bias;
        FloatArray routed_gate_up_proj;
        FloatArray routed_down_proj;
        FloatArray shared_gate_proj;
        FloatArray shared_up_proj;
        FloatArray shared_down_proj;
        FloatArray router_logits_expected;
        IntArray topk_indices_expected;
        FloatArray topk_weights_expected;
        FloatArray routed_out_expected;
        FloatArray shared_out_expected;
        FloatArray final_out_expected;
        float *router_logits_actual;
        int *topk_indices_actual;
        float *topk_weights_actual;
        float *routed_out_actual;
        float *shared_out_actual;
        float *final_out_actual;
        int ok;

        if (next_case != NULL) {
            saved_char = *next_case;
            *next_case = '\0';
        }
        hidden_states = parse_float_array_after_key(cursor, "hidden_states");
        router_weight = parse_float_array_after_key(cursor, "router_weight");
        e_score_correction_bias = parse_float_array_after_key(cursor, "e_score_correction_bias");
        routed_gate_up_proj = parse_float_array_after_key(cursor, "routed_gate_up_proj");
        routed_down_proj = parse_float_array_after_key(cursor, "routed_down_proj");
        shared_gate_proj = parse_float_array_after_key(cursor, "shared_gate_proj");
        shared_up_proj = parse_float_array_after_key(cursor, "shared_up_proj");
        shared_down_proj = parse_float_array_after_key(cursor, "shared_down_proj");
        router_logits_expected = parse_float_array_after_key(cursor, "router_logits");
        topk_indices_expected = parse_int_array_after_key(cursor, "topk_indices");
        topk_weights_expected = parse_float_array_after_key(cursor, "topk_weights");
        routed_out_expected = parse_float_array_after_key(cursor, "routed_out");
        shared_out_expected = parse_float_array_after_key(cursor, "shared_out");
        final_out_expected = parse_float_array_after_key(cursor, "final_out");
        if (next_case != NULL) {
            *next_case = saved_char;
        }

        router_logits_actual = (float *)malloc((size_t)router_logits_expected.length * sizeof(float));
        topk_indices_actual = (int *)malloc((size_t)topk_indices_expected.length * sizeof(int));
        topk_weights_actual = (float *)malloc((size_t)topk_weights_expected.length * sizeof(float));
        routed_out_actual = (float *)malloc((size_t)routed_out_expected.length * sizeof(float));
        shared_out_actual = (float *)malloc((size_t)shared_out_expected.length * sizeof(float));
        final_out_actual = (float *)malloc((size_t)final_out_expected.length * sizeof(float));

        deepseek_moe_forward(
            hidden_states.data,
            batch_size,
            seq_len,
            &config,
            router_weight.data,
            e_score_correction_bias.data,
            routed_gate_up_proj.data,
            routed_down_proj.data,
            shared_gate_proj.data,
            shared_up_proj.data,
            shared_down_proj.data,
            router_logits_actual,
            topk_indices_actual,
            topk_weights_actual,
            routed_out_actual,
            shared_out_actual,
            final_out_actual
        );

        ok = compare_float_arrays(router_logits_actual, router_logits_expected.data, router_logits_expected.length, "moe_router_logits") &&
             compare_int_arrays(topk_indices_actual, topk_indices_expected.data, topk_indices_expected.length, "moe_topk_indices") &&
             compare_float_arrays(topk_weights_actual, topk_weights_expected.data, topk_weights_expected.length, "moe_topk_weights") &&
             compare_float_arrays(routed_out_actual, routed_out_expected.data, routed_out_expected.length, "moe_routed_out") &&
             compare_float_arrays(shared_out_actual, shared_out_expected.data, shared_out_expected.length, "moe_shared_out") &&
             compare_float_arrays(final_out_actual, final_out_expected.data, final_out_expected.length, "moe_final_out");
        if (!ok) {
            printf("FAIL moe case_%d\n", case_count);
            free_float_array(hidden_states);
            free_float_array(router_weight);
            free_float_array(e_score_correction_bias);
            free_float_array(routed_gate_up_proj);
            free_float_array(routed_down_proj);
            free_float_array(shared_gate_proj);
            free_float_array(shared_up_proj);
            free_float_array(shared_down_proj);
            free_float_array(router_logits_expected);
            free_int_array(topk_indices_expected);
            free_float_array(topk_weights_expected);
            free_float_array(routed_out_expected);
            free_float_array(shared_out_expected);
            free_float_array(final_out_expected);
            free(router_logits_actual);
            free(topk_indices_actual);
            free(topk_weights_actual);
            free(routed_out_actual);
            free(shared_out_actual);
            free(final_out_actual);
            free(json);
            return 1;
        }

        cases_passed += 1;
        case_count += 1;
        free_float_array(hidden_states);
        free_float_array(router_weight);
        free_float_array(e_score_correction_bias);
        free_float_array(routed_gate_up_proj);
        free_float_array(routed_down_proj);
        free_float_array(shared_gate_proj);
        free_float_array(shared_up_proj);
        free_float_array(shared_down_proj);
        free_float_array(router_logits_expected);
        free_int_array(topk_indices_expected);
        free_float_array(topk_weights_expected);
        free_float_array(routed_out_expected);
        free_float_array(shared_out_expected);
        free_float_array(final_out_expected);
        free(router_logits_actual);
        free(topk_indices_actual);
        free(topk_weights_actual);
        free(routed_out_actual);
        free(shared_out_actual);
        free(final_out_actual);
        cursor = next_case != NULL ? next_case : cursor + 6;
    }

    printf("PASS moe %d cases\n", cases_passed);
    free(json);
    return 0;
}

int main(void) {
    if (run_mlp_cases("../test_cases/mlp_cases.json") != 0) {
        return 1;
    }
    if (run_router_cases("../test_cases/router_cases.json") != 0) {
        return 1;
    }
    if (run_route_cases("../test_cases/route_cases.json") != 0) {
        return 1;
    }
    if (run_naive_moe_cases("../test_cases/naive_moe_cases.json") != 0) {
        return 1;
    }
    if (run_moe_cases("../test_cases/moe_cases.json") != 0) {
        return 1;
    }
    return 0;
}
