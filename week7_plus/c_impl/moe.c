#include "moe.h"

#include <math.h>
#include <float.h>
#include <stdlib.h>

float deepseek_silu(float x) {
    return x / (1.0f + expf(-x));
}

void deepseek_mlp_forward(
    const float *hidden_states,
    int num_tokens,
    int hidden_size,
    int intermediate_size,
    const float *gate_proj_weight,
    const float *up_proj_weight,
    const float *down_proj_weight,
    float *output
) {
    float *gate = (float *)malloc((size_t)intermediate_size * sizeof(float));
    float *up = (float *)malloc((size_t)intermediate_size * sizeof(float));
    float *mixed = (float *)malloc((size_t)intermediate_size * sizeof(float));
    int token_idx;
    int out_idx;
    int in_idx;

    for (token_idx = 0; token_idx < num_tokens; ++token_idx) {
        const float *token = hidden_states + (size_t)token_idx * (size_t)hidden_size;
        float *token_out = output + (size_t)token_idx * (size_t)hidden_size;

        for (out_idx = 0; out_idx < intermediate_size; ++out_idx) {
            float gate_acc = 0.0f;
            float up_acc = 0.0f;
            for (in_idx = 0; in_idx < hidden_size; ++in_idx) {
                gate_acc += gate_proj_weight[(size_t)out_idx * (size_t)hidden_size + (size_t)in_idx] * token[in_idx];
                up_acc += up_proj_weight[(size_t)out_idx * (size_t)hidden_size + (size_t)in_idx] * token[in_idx];
            }
            gate[out_idx] = deepseek_silu(gate_acc);
            up[out_idx] = up_acc;
            mixed[out_idx] = gate[out_idx] * up[out_idx];
        }

        for (out_idx = 0; out_idx < hidden_size; ++out_idx) {
            float down_acc = 0.0f;
            for (in_idx = 0; in_idx < intermediate_size; ++in_idx) {
                down_acc += down_proj_weight[(size_t)out_idx * (size_t)intermediate_size + (size_t)in_idx] * mixed[in_idx];
            }
            token_out[out_idx] = down_acc;
        }
    }

    free(gate);
    free(up);
    free(mixed);
}

void deepseek_topk_router_forward(
    const float *hidden_states,
    int num_tokens,
    int hidden_size,
    int n_routed_experts,
    const float *router_weight,
    float *router_logits
) {
    int token_idx;
    int expert_idx;
    int hidden_idx;

    for (token_idx = 0; token_idx < num_tokens; ++token_idx) {
        const float *token = hidden_states + (size_t)token_idx * (size_t)hidden_size;
        float *token_logits = router_logits + (size_t)token_idx * (size_t)n_routed_experts;

        for (expert_idx = 0; expert_idx < n_routed_experts; ++expert_idx) {
            float acc = 0.0f;
            for (hidden_idx = 0; hidden_idx < hidden_size; ++hidden_idx) {
                acc += router_weight[(size_t)expert_idx * (size_t)hidden_size + (size_t)hidden_idx] * token[hidden_idx];
            }
            token_logits[expert_idx] = acc;
        }
    }
}

static float sigmoidf32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void deepseek_route_tokens_to_experts(
    const float *router_logits,
    int num_tokens,
    int n_routed_experts,
    int n_group,
    int topk_group,
    int top_k,
    int norm_topk_prob,
    float routed_scaling_factor,
    const float *e_score_correction_bias,
    int *topk_indices,
    float *topk_weights
) {
    int experts_per_group = n_routed_experts / n_group;
    float *router_probs = (float *)malloc((size_t)n_routed_experts * sizeof(float));
    float *router_choice = (float *)malloc((size_t)n_routed_experts * sizeof(float));
    float *group_scores = (float *)malloc((size_t)n_group * sizeof(float));
    int *selected_groups = (int *)malloc((size_t)topk_group * sizeof(int));
    int token_idx;
    int expert_idx;
    int group_idx;
    int top_idx;

    for (token_idx = 0; token_idx < num_tokens; ++token_idx) {
        const float *token_logits = router_logits + (size_t)token_idx * (size_t)n_routed_experts;
        int *token_topk_indices = topk_indices + (size_t)token_idx * (size_t)top_k;
        float *token_topk_weights = topk_weights + (size_t)token_idx * (size_t)top_k;
        float weight_sum = 0.0f;

        for (expert_idx = 0; expert_idx < n_routed_experts; ++expert_idx) {
            router_probs[expert_idx] = sigmoidf32(token_logits[expert_idx]);
            router_choice[expert_idx] = router_probs[expert_idx] + e_score_correction_bias[expert_idx];
        }

        for (group_idx = 0; group_idx < n_group; ++group_idx) {
            float best = -FLT_MAX;
            float second = -FLT_MAX;
            for (expert_idx = 0; expert_idx < experts_per_group; ++expert_idx) {
                float value = router_choice[group_idx * experts_per_group + expert_idx];
                if (value > best) {
                    second = best;
                    best = value;
                } else if (value > second) {
                    second = value;
                }
            }
            group_scores[group_idx] = best + second;
        }

        for (top_idx = 0; top_idx < topk_group; ++top_idx) {
            float best = -FLT_MAX;
            int best_group = 0;
            for (group_idx = 0; group_idx < n_group; ++group_idx) {
                int already_selected = 0;
                int selected_idx;
                for (selected_idx = 0; selected_idx < top_idx; ++selected_idx) {
                    if (selected_groups[selected_idx] == group_idx) {
                        already_selected = 1;
                        break;
                    }
                }
                if (!already_selected && group_scores[group_idx] > best) {
                    best = group_scores[group_idx];
                    best_group = group_idx;
                }
            }
            selected_groups[top_idx] = best_group;
        }

        for (top_idx = 0; top_idx < top_k; ++top_idx) {
            float best = -FLT_MAX;
            int best_expert = 0;
            for (expert_idx = 0; expert_idx < n_routed_experts; ++expert_idx) {
                int group = expert_idx / experts_per_group;
                int in_selected_group = 0;
                int not_selected_yet = 1;
                float score_for_choice;
                int selected_idx;
                for (selected_idx = 0; selected_idx < topk_group; ++selected_idx) {
                    if (selected_groups[selected_idx] == group) {
                        in_selected_group = 1;
                        break;
                    }
                }
                score_for_choice = in_selected_group ? router_choice[expert_idx] : 0.0f;
                for (selected_idx = 0; selected_idx < top_idx; ++selected_idx) {
                    if (token_topk_indices[selected_idx] == expert_idx) {
                        not_selected_yet = 0;
                        break;
                    }
                }
                if (not_selected_yet && score_for_choice > best) {
                    best = score_for_choice;
                    best_expert = expert_idx;
                }
            }
            token_topk_indices[top_idx] = best_expert;
            token_topk_weights[top_idx] = router_probs[best_expert];
            weight_sum += token_topk_weights[top_idx];
        }

        if (norm_topk_prob) {
            for (top_idx = 0; top_idx < top_k; ++top_idx) {
                token_topk_weights[top_idx] = token_topk_weights[top_idx] / (weight_sum + 1e-20f);
            }
        }
        for (top_idx = 0; top_idx < top_k; ++top_idx) {
            token_topk_weights[top_idx] *= routed_scaling_factor;
        }
    }

    free(router_probs);
    free(router_choice);
    free(group_scores);
    free(selected_groups);
}

void deepseek_naive_moe_forward(
    const float *hidden_states,
    int num_tokens,
    int hidden_size,
    int num_local_experts,
    int intermediate_size,
    int top_k,
    const int *top_k_index,
    const float *top_k_weights,
    const float *gate_up_proj,
    const float *down_proj,
    float *output
) {
    float *gate = (float *)malloc((size_t)intermediate_size * sizeof(float));
    float *up = (float *)malloc((size_t)intermediate_size * sizeof(float));
    float *mixed = (float *)malloc((size_t)intermediate_size * sizeof(float));
    int token_idx;
    int hidden_idx;

    for (token_idx = 0; token_idx < num_tokens * hidden_size; ++token_idx) {
        output[token_idx] = 0.0f;
    }

    for (token_idx = 0; token_idx < num_tokens; ++token_idx) {
        const float *token = hidden_states + (size_t)token_idx * (size_t)hidden_size;
        float *token_out = output + (size_t)token_idx * (size_t)hidden_size;
        int top_idx;

        for (top_idx = 0; top_idx < top_k; ++top_idx) {
            int expert_idx = top_k_index[(size_t)token_idx * (size_t)top_k + (size_t)top_idx];
            float expert_weight = top_k_weights[(size_t)token_idx * (size_t)top_k + (size_t)top_idx];
            const float *expert_gate_up = gate_up_proj + (size_t)expert_idx * (size_t)(2 * intermediate_size) * (size_t)hidden_size;
            const float *expert_down = down_proj + (size_t)expert_idx * (size_t)hidden_size * (size_t)intermediate_size;
            int inter_idx;

            if (expert_idx < 0 || expert_idx >= num_local_experts) {
                continue;
            }

            for (inter_idx = 0; inter_idx < intermediate_size; ++inter_idx) {
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (hidden_idx = 0; hidden_idx < hidden_size; ++hidden_idx) {
                    gate_acc += expert_gate_up[(size_t)inter_idx * (size_t)hidden_size + (size_t)hidden_idx] * token[hidden_idx];
                    up_acc += expert_gate_up[(size_t)(inter_idx + intermediate_size) * (size_t)hidden_size + (size_t)hidden_idx] * token[hidden_idx];
                }
                gate[inter_idx] = deepseek_silu(gate_acc);
                up[inter_idx] = up_acc;
                mixed[inter_idx] = gate[inter_idx] * up[inter_idx];
            }

            for (hidden_idx = 0; hidden_idx < hidden_size; ++hidden_idx) {
                float down_acc = 0.0f;
                for (inter_idx = 0; inter_idx < intermediate_size; ++inter_idx) {
                    down_acc += expert_down[(size_t)hidden_idx * (size_t)intermediate_size + (size_t)inter_idx] * mixed[inter_idx];
                }
                token_out[hidden_idx] += down_acc * expert_weight;
            }
        }
    }

    free(gate);
    free(up);
    free(mixed);
}

void deepseek_moe_forward(
    const float *hidden_states,
    int batch_size,
    int seq_len,
    const DeepseekMoeConfig *config,
    const float *router_weight,
    const float *e_score_correction_bias,
    const float *routed_gate_up_proj,
    const float *routed_down_proj,
    const float *shared_gate_proj,
    const float *shared_up_proj,
    const float *shared_down_proj,
    float *router_logits,
    int *topk_indices,
    float *topk_weights,
    float *routed_out,
    float *shared_out,
    float *final_out
) {
    int num_tokens = batch_size * seq_len;
    int hidden_size = config->hidden_size;
    int shared_intermediate_size = config->moe_intermediate_size * config->n_shared_experts;
    int idx;

    deepseek_topk_router_forward(
        hidden_states,
        num_tokens,
        hidden_size,
        config->n_routed_experts,
        router_weight,
        router_logits
    );
    deepseek_route_tokens_to_experts(
        router_logits,
        num_tokens,
        config->n_routed_experts,
        config->n_group,
        config->topk_group,
        config->num_experts_per_tok,
        config->norm_topk_prob,
        config->routed_scaling_factor,
        e_score_correction_bias,
        topk_indices,
        topk_weights
    );
    deepseek_naive_moe_forward(
        hidden_states,
        num_tokens,
        hidden_size,
        config->num_local_experts,
        config->moe_intermediate_size,
        config->num_experts_per_tok,
        topk_indices,
        topk_weights,
        routed_gate_up_proj,
        routed_down_proj,
        routed_out
    );
    deepseek_mlp_forward(
        hidden_states,
        num_tokens,
        hidden_size,
        shared_intermediate_size,
        shared_gate_proj,
        shared_up_proj,
        shared_down_proj,
        shared_out
    );
    for (idx = 0; idx < num_tokens * hidden_size; ++idx) {
        final_out[idx] = routed_out[idx] + shared_out[idx];
    }
}
