#ifndef DEEPSEEK_MOE_H
#define DEEPSEEK_MOE_H

#include <stddef.h>

typedef struct DeepseekMoeConfig {
    int hidden_size;
    int intermediate_size;
    int moe_intermediate_size;
    int n_routed_experts;
    int num_local_experts;
    int n_shared_experts;
    int n_group;
    int topk_group;
    int num_experts_per_tok;
    int norm_topk_prob;
    float routed_scaling_factor;
} DeepseekMoeConfig;

float deepseek_silu(float x);

void deepseek_mlp_forward(
    const float *hidden_states,
    int num_tokens,
    int hidden_size,
    int intermediate_size,
    const float *gate_proj_weight,
    const float *up_proj_weight,
    const float *down_proj_weight,
    float *output
);

void deepseek_topk_router_forward(
    const float *hidden_states,
    int num_tokens,
    int hidden_size,
    int n_routed_experts,
    const float *router_weight,
    float *router_logits
);

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
);

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
);

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
);

#endif
