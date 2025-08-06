import argparse
import os
import torch


def convert_hf_to_megatron_minicpm_main(
    load_path: str,
    num_layer: int = 52,
    tp_size: int = 1,
    tp_rank: int = 0,
    pp_size: int = 1,
    pp_rank: int = 0,
    save_dir: str = "./tmp",
    num_kv_heads: int = 8,
    num_query_heads: int = 24,
    use_mla: bool = False,
):
    """
    Main conversion function for HF to MiniCPM Megatron conversion

    Args:
        load_path: Path to the HF weights file
        num_layer: Number of transformer layers
        tp_size: Tensor parallel size
        tp_rank: Tensor parallel rank
        pp_size: Pipeline parallel size
        pp_rank: Pipeline parallel rank
        save_dir: Directory to save the Megatron checkpoint
        num_kv_heads: Number of KV attention heads
        num_query_heads: Number of query attention heads
        use_mla: Whether to use MLA (Multi-head Latent Attention)
    """
    if pp_size != 1:
        output_path = save_dir + "/" + f"mp_rank_0{tp_rank}_00{pp_rank}/"
    else:
        output_path = save_dir + "/" + f"mp_rank_0{tp_rank}"
    os.makedirs(output_path, exist_ok=True)

    cpm_model_dict = torch.load(load_path)
    # llama_model = AutoModelForCausalLM.from_pretrained(load_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    # cpm_model_dict = llama_model.state_dict()

    for k in cpm_model_dict:
        print(k, cpm_model_dict[k].shape, cpm_model_dict[k].dtype)

    megatron_model_dict = dict()

    embedding = cpm_model_dict["model.embed_tokens.weight"]
    assert embedding.shape[0] % tp_size == 0
    if pp_rank == 0:
        megatron_model_dict["embedding.word_embeddings.weight"] = torch.split(
            embedding, embedding.shape[0] // tp_size, dim=0
        )[tp_rank]
    if pp_rank == pp_size - 1:
        megatron_model_dict["decoder.final_layernorm.weight"] = cpm_model_dict["model.norm.weight"]

    assert num_query_heads % num_kv_heads == 0
    assert num_layer % pp_size == 0

    layer_num_per_pp = num_layer // pp_size
    num_query_heads_per_group = num_query_heads // num_kv_heads

    if use_mla:
        for layer_idx in range(pp_rank * layer_num_per_pp, (pp_rank + 1) * layer_num_per_pp):
            print(f"process layer {layer_idx}")
            layer_idx_abs = layer_idx - pp_rank * layer_num_per_pp

            megatron_model_dict[f"decoder.layers.{layer_idx_abs}.input_layernorm.weight"] = cpm_model_dict[
                f"model.layers.{layer_idx}.input_layernorm.weight"
            ].clone()
            q_b_proj_weight = cpm_model_dict[f"model.layers.{layer_idx}.self_attn.q_b_proj.weight"]
            megatron_model_dict[f"decoder.layers.{layer_idx}.self_attention.linear_q_up.weight"] = torch.split(
                q_b_proj_weight, dim=0, split_size_or_sections=q_b_proj_weight.shape[0] // tp_size
            )[tp_rank].clone()

            q_a_proj_weight = cpm_model_dict[f"model.layers.{layer_idx}.self_attn.q_a_proj.weight"]
            megatron_model_dict[f"decoder.layers.{layer_idx}.self_attention.linear_q_down.weight"] = torch.split(
                q_a_proj_weight, dim=0, split_size_or_sections=q_a_proj_weight.shape[0] // tp_size
            )[tp_rank].clone()

            kv_b_proj_weight = cpm_model_dict[f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight"]
            megatron_model_dict[f"decoder.layers.{layer_idx}.self_attention.linear_kv_up.weight"] = torch.split(
                kv_b_proj_weight, dim=0, split_size_or_sections=kv_b_proj_weight.shape[0] // tp_size
            )[tp_rank].clone()

            kv_a_proj_weight = cpm_model_dict[f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight"]
            megatron_model_dict[f"decoder.layers.{layer_idx}.self_attention.linear_kv_down.weight"] = torch.split(
                kv_a_proj_weight, dim=0, split_size_or_sections=kv_a_proj_weight.shape[0] // tp_size
            )[tp_rank].clone()

            megatron_model_dict[f"decoder.layers.{layer_idx}.self_attention.q_layernorm.weight"] = cpm_model_dict[
                f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight"
            ]
            megatron_model_dict[f"decoder.layers.{layer_idx}.self_attention.kv_layernorm.weight"] = cpm_model_dict[
                f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight"
            ]
            o_weight = cpm_model_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"]
            megatron_model_dict[f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight"] = torch.split(
                o_weight, dim=1, split_size_or_sections=o_weight.shape[1] // tp_size
            )[tp_rank].clone()

            megatron_model_dict[f"decoder.layers.{layer_idx}.pre_mlp_layernorm.weight"] = cpm_model_dict[
                f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            ].clone()
            w0_weight = cpm_model_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
            w1_weight = cpm_model_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
            w2_weight = cpm_model_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"]
            w0_weight_cu_rank = torch.split(w0_weight, w0_weight.shape[0] // tp_size, dim=0)[tp_rank]
            w1_weight_cu_rank = torch.split(w1_weight, w1_weight.shape[0] // tp_size, dim=0)[tp_rank]
            megatron_model_dict[f"decoder.layers.{layer_idx_abs}.mlp.linear_fc1.weight"] = torch.cat(
                [w0_weight_cu_rank, w1_weight_cu_rank], dim=0
            ).clone()
            megatron_model_dict[f"decoder.layers.{layer_idx_abs}.mlp.linear_fc1._extra_state"] = None
            megatron_model_dict[f"decoder.layers.{layer_idx_abs}.mlp.linear_fc2.weight"] = torch.split(
                w2_weight, w2_weight.shape[1] // tp_size, dim=1
            )[tp_rank].clone()
            megatron_model_dict[f"decoder.layers.{layer_idx_abs}.mlp.linear_fc2._extra_state"] = None
    else:
        for layer_idx in range(pp_rank * layer_num_per_pp, (pp_rank + 1) * layer_num_per_pp):
            print(f"process layer {layer_idx}")
            layer_idx_abs = layer_idx - pp_rank * layer_num_per_pp
            q_weight = cpm_model_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"]
            k_weight = cpm_model_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"]
            v_weight = cpm_model_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"]
            o_weight = cpm_model_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"]
            # print(q_weight.shape)
            q_weight_split = torch.split(q_weight, dim=0, split_size_or_sections=64)
            k_weight_split = torch.split(k_weight, dim=0, split_size_or_sections=64)
            v_weight_split = torch.split(v_weight, dim=0, split_size_or_sections=64)
            print(len(q_weight_split), q_weight_split[0].shape, len(k_weight_split))
            qkv_weight_list = []
            for i in range(num_kv_heads):
                q_group_weight = []
                for j in range(num_query_heads_per_group):
                    # print(i * num_query_heads_per_group + j)
                    q_group_weight.append(q_weight_split[i * num_query_heads_per_group + j])
                qkv_weight_list.extend(q_group_weight)
                qkv_weight_list.extend([k_weight_split[i], v_weight_split[i]])
            qkv_weight = torch.cat(qkv_weight_list, dim=0)
            qkv_weight_tp = torch.split(qkv_weight, qkv_weight.shape[0] // tp_size, dim=0)
            megatron_model_dict[f"decoder.layers.{layer_idx_abs}.self_attention.linear_qkv.weight"] = qkv_weight_tp[
                tp_rank
            ]
            megatron_model_dict[f"decoder.layers.{layer_idx_abs}.self_attention.linear_qkv._extra_state"] = None
            megatron_model_dict[f"decoder.layers.{layer_idx_abs}.self_attention.linear_proj.weight"] = torch.split(
                o_weight, o_weight.shape[1] // tp_size, dim=1
            )[tp_rank]
            megatron_model_dict[f"decoder.layers.{layer_idx_abs}.self_attention.linear_proj._extra_state"] = None
            megatron_model_dict[f"decoder.layers.{layer_idx_abs}.self_attention.linear_qkv.layer_norm_weight"] = (
                cpm_model_dict[f"model.layers.{layer_idx}.input_layernorm.weight"]
            )

            megatron_model_dict[f"decoder.layers.{layer_idx_abs}.mlp.linear_fc1.layer_norm_weight"] = cpm_model_dict[
                f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            ]
            w0_weight = cpm_model_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
            w1_weight = cpm_model_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
            w2_weight = cpm_model_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"]
            w0_weight_cu_rank = torch.split(w0_weight, w0_weight.shape[0] // tp_size, dim=0)[tp_rank]
            w1_weight_cu_rank = torch.split(w1_weight, w1_weight.shape[0] // tp_size, dim=0)[tp_rank]
            megatron_model_dict[f"decoder.layers.{layer_idx_abs}.mlp.linear_fc1.weight"] = torch.cat(
                [w0_weight_cu_rank, w1_weight_cu_rank], dim=0
            )
            megatron_model_dict[f"decoder.layers.{layer_idx_abs}.mlp.linear_fc1._extra_state"] = None
            megatron_model_dict[f"decoder.layers.{layer_idx_abs}.mlp.linear_fc2.weight"] = torch.split(
                w2_weight, w2_weight.shape[1] // tp_size, dim=1
            )[tp_rank]
            megatron_model_dict[f"decoder.layers.{layer_idx_abs}.mlp.linear_fc2._extra_state"] = None

    megatron_ckpt = {"model": megatron_model_dict}

    num_params = 0
    for key in megatron_model_dict:
        if "extra_state" not in key:
            num_params += megatron_model_dict[key].numel()
            # print(megatron_model_dict[key].dtype)

    torch.save(megatron_ckpt, output_path + "/model_optim_rng.pt")
    print("num params: ", num_params)


def convert_hf_to_megatron_minicpm(
    checkpoint_path: str,
    output_path: str,
    num_layer: int,
    tp_size: int,
    pp_size: int,
    num_kv_heads: int,
    num_query_heads: int,
    **kwargs,
) -> None:
    """
    Wrapper function for HF to MiniCPM Megatron conversion

    Args:
        checkpoint_path: Path to the HF weights file
        output_path: Path to save the Megatron checkpoint directory
        num_layer: Number of transformer layers
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
        num_kv_heads: Number of KV attention heads
        num_query_heads: Number of query attention heads
        **kwargs: Additional arguments for the conversion
    """
    print(f"Converting HF to MiniCPM Megatron: {checkpoint_path} -> {output_path}")
    print(f"Model config: {num_layer} layers, TP={tp_size}, PP={pp_size}")
    print(f"Attention config: {num_query_heads} query heads, {num_kv_heads} KV heads")

    # This would need to be implemented to call the main conversion logic
    # For now, it's a placeholder that can be extended based on the specific needs
    pass


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", required=True, help="Path to HF weights file")
    parser.add_argument("--num_layer", type=int, default=52, help="Number of transformer layers")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--tp_rank", type=int, default=0, help="Tensor parallel rank")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--pp_rank", type=int, default=0, help="Pipeline parallel rank")
    parser.add_argument("--save_dir", type=str, default="./tmp", help="Directory to save Megatron checkpoint")
    parser.add_argument("--num_kv_heads", type=int, default=8, help="Number of KV attention heads")
    parser.add_argument("--num_query_heads", type=int, default=24, help="Number of query attention heads")
    parser.add_argument("--use_mla", action="store_true", help="Use MLA (Multi-head Latent Attention)")
    args = parser.parse_args()

    convert_hf_to_megatron_minicpm_main(
        load_path=args.load,
        num_layer=args.num_layer,
        tp_size=args.tp_size,
        tp_rank=args.tp_rank,
        pp_size=args.pp_size,
        pp_rank=args.pp_rank,
        save_dir=args.save_dir,
        num_kv_heads=args.num_kv_heads,
        num_query_heads=args.num_query_heads,
        use_mla=args.use_mla,
    )


if __name__ == "__main__":
    main()
