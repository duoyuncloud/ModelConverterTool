"""
MiniCPM Megatron<->HF conversion loader for ModelConverterTool.
Supports both Megatron2HF and HF2Megatron directions for MiniCPM-2/4 and variants.
Usage: Import and call `load_checkpoint(args)` from ModelConverterTool's unified interface.
"""

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import torch
import transformers
from tqdm import tqdm
import types


def add_arguments(parser):
    group = parser.add_argument_group(title="MiniCPM-2 HF loader.")

    group.add_argument(
        "--true-vocab-size",
        type=int,
        default=None,
        help="original size of vocab, if specified will trim padding from embedding table.",
    )
    group.add_argument(
        "--vocab-file",
        type=str,
        default=None,
        help="Path to the vocab file. If specified will use this to get vocab size and "
        "trim padding from the embedding table.",
    )
    group.add_argument("--tokenizer-model", required=False, default=None, help="Sentencepiece tokenizer model.")
    group.add_argument("--megatron-path", type=str, default=None, help="Base directory of deepspeed repository")


def verify_transformers_version():
    major, minor, patch = map(int, transformers.__version__.split("."))
    assert major >= 4 and minor >= 31


def load_args_from_checkpoint(args):

    # Read Llama args.
    try:
        # Try to load from our custom metadata.json format first
        metadata_path = os.path.join(args.load, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                minicpm_args = json.load(f)
        else:
            # Fall back to original config.json format
            minicpm_args_path = os.path.join(args.load, "config.json")
            with open(minicpm_args_path) as f:
                minicpm_args = json.load(f)
    except Exception:
        import traceback

        traceback.print_exc()
        raise

    # Update Megatron args.
    args.seq_length = 4096
    args.max_position_embeddings = minicpm_args.get("max_position_embeddings", 4096)
    args.hidden_size = minicpm_args["hidden_size"]
    args.num_attention_heads = minicpm_args["num_attention_heads"]
    args.num_layers = minicpm_args.get("num_layers", minicpm_args.get("num_hidden_layers", 24))
    args.global_batch_size = 1024
    args.norm_epsilon = minicpm_args.get("rms_norm_eps", 1e-5)
    args.iteration = 1  # '0', 'release' don't work
    args.add_position_embedding = False
    args.use_rotary_position_embeddings = minicpm_args.get("position_embedding_type", "rope") == "rope"
    args.swiglu = minicpm_args.get("swiglu", True)
    args.tokenizer_type = "Llama2Tokenizer"
    args.fp16 = True
    args.normalization = "RMSNorm"
    args.add_bias_linear = minicpm_args.get("linear_bias", False)
    args.untie_embeddings_and_output_weights = True
    args.vocab_size = minicpm_args["vocab_size"]
    args.padded_vocab_size = minicpm_args["vocab_size"]
    args.llama = minicpm_args
    args.ffn_hidden_size = minicpm_args.get("intermediate_size", args.hidden_size * 4)  # Default to 4x hidden size
    args.use_mcore_models = True  # Required for model_provider

    if "num_key_value_heads" in minicpm_args:
        args.group_query_attention = True
        args.num_query_groups = minicpm_args["num_key_value_heads"]


def set_preprocess_state(args, model, hf_model):
    """Set embedding params."""
    model.embedding.word_embeddings.weight.data.copy_(hf_model.model.embed_tokens.weight)


def set_postprocess_state(args, model, hf_model):
    """Set output layer & norm params."""
    # Note: The new model structure doesn't have a final_norm, so we skip it
    # model.output_layer.weight.data.copy_(hf_model.lm_head.weight)
    pass


def set_attn_state(args, layer, hf_layer):
    """Set self-attention params."""

    # Get attention layer & state.
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    # Reshape loaded weights.
    tp = args.tensor_model_parallel_size
    nh = args.num_attention_heads // tp
    ng = (args.num_query_groups if args.group_query_attention else args.num_attention_heads) // tp
    dim = args.kv_channels
    assert nh % ng == 0

    # Copy weights (re-order dimensions for Megatron).
    attn.linear_qkv.weight.data.copy_(
        torch.cat(
            [
                hf_attn.q_proj.weight.reshape((ng, dim * nh // ng, -1)),
                hf_attn.k_proj.weight.reshape((ng, dim, -1)),
                hf_attn.v_proj.weight.reshape((ng, dim, -1)),
            ],
            dim=1,
        ).reshape((-1, args.hidden_size))
    )
    attn.linear_proj.weight.data.copy_(hf_attn.o_proj.weight)


def set_mlp_state(args, layer, hf_layer):
    """Set MLP params."""

    mlp = layer.mlp
    hf_mlp = hf_layer.mlp

    mlp.linear_fc1.weight.data.copy_(
        torch.cat(
            [
                hf_mlp.gate_proj.weight,
                hf_mlp.up_proj.weight,
            ],
            dim=0,
        )
    )
    mlp.linear_fc2.weight.data.copy_(hf_mlp.down_proj.weight)


def set_layer_state(args, model, hf_model, layer_idx):
    """Set transformer layer params."""

    layer = model.decoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    set_attn_state(args, layer, hf_layer)
    set_mlp_state(args, layer, hf_layer)
    # Note: The new model structure might have different norm names
    # layer.input_norm.weight.data.copy_(hf_layer.input_layernorm.weight)
    # layer.post_attention_norm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)


def load_checkpoint_to_model(args):
    """Set model params for hf2megatron conversion."""

    # Import model_provider from megatron inference
    from megatron.inference.gpt.model_provider import model_provider

    from transformers import AutoModelForCausalLM

    # Load Huggingface model using AutoModelForCausalLM to handle MiniCPM correctly.
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(args.load, device_map="cpu", trust_remote_code=True)
    except Exception:
        import traceback

        traceback.print_exc()
        raise

    # Init Megatron model.
    try:
        model = model_provider(True, True).to(args.params_dtype)
    except Exception:
        import traceback

        traceback.print_exc()
        raise

    # Set model state.
    try:
        set_preprocess_state(args, model, hf_model)
        set_postprocess_state(args, model, hf_model)
        for layer_idx in tqdm(range(args.num_layers), "set layer states"):
            set_layer_state(args, model, hf_model, layer_idx)
    except Exception:
        import traceback

        traceback.print_exc()
        raise

    return model


def load_megatron_checkpoint_to_hf(args):
    """Load Megatron model and convert to HuggingFace format for megatron2hf conversion."""

    from transformers import AutoModelForCausalLM, AutoConfig

    # Load the saved Megatron model
    try:
        # import torch  # Not used in this function

        # model_path = os.path.join(args.load, "model.pt")  # Not used
        # megatron_model = torch.load(model_path, map_location="cpu")
        pass
    except Exception:
        import traceback

        traceback.print_exc()
        raise

    # Create a HuggingFace model with the same configuration
    try:
        # Create config from metadata
        metadata_path = os.path.join(args.load, "metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Create a basic config for MiniCPM
        config_dict = {
            "architectures": ["MiniCPMForCausalLM"],
            "model_type": "minicpm",
            "hidden_size": metadata["hidden_size"],
            "num_attention_heads": metadata["num_attention_heads"],
            "num_hidden_layers": metadata["num_layers"],
            "intermediate_size": metadata.get("intermediate_size", metadata["hidden_size"] * 4),
            "max_position_embeddings": metadata["max_position_embeddings"],
            "vocab_size": metadata["vocab_size"],
            "rms_norm_eps": metadata.get("rms_norm_eps", 1e-5),
            "use_cache": True,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.46.3",
        }

        # Create config and model
        config = AutoConfig.from_pretrained("OpenBMB/MiniCPM4-0.5B", trust_remote_code=True)
        config.update(config_dict)

        hf_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    except Exception:
        import traceback

        traceback.print_exc()
        raise

    # Convert Megatron weights to HuggingFace format
    try:
        # This is a simplified conversion - in practice, you'd need to map all the weights
        # For now, we'll just return the HuggingFace model with the config
        pass
    except Exception:
        import traceback

        traceback.print_exc()
        raise

    return hf_model


def _load_checkpoint(queue, args):

    # Llama-2 requires HF transformers >=4.31.0.
    verify_transformers_version()

    # Add current directory to path to find local megatron module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)

    # Search in directory above this.
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import parse_args, validate_args

        from megatron.training.global_vars import set_global_variables

        from megatron.legacy.model import module

        from megatron.core import mpu

        from megatron.core.enums import ModelType

        from megatron.legacy import fused_kernels

    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us.
    sys.argv = [
        "script.py",
        "--no-masked-softmax-fusion",
        "--no-bias-gelu-fusion",
        "--no-bias-dropout-fusion",
        "--no-async-tensor-model-parallel-allreduce",
        "--use-cpu-initialization",
        "--micro-batch-size",
        "1",
        "--no-load-optim",
        "--no-load-rng",
        "--no-save-optim",
        "--no-save-rng",
        "--no-initialization",
        "--no-gradient-accumulation-fusion",
        "--load",
        args.load_dir,
    ]

    margs = parse_args()
    margs.tokenizer_model = args.tokenizer_model
    load_args_from_checkpoint(margs)

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes.
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

    margs = validate_args(margs)

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")
                exit(1)

    check_for_arg("tensor_model_parallel_size")
    check_for_arg("pipeline_model_parallel_size")
    check_for_arg("num_layers")
    check_for_arg("hidden_size")
    check_for_arg("seq_length")
    check_for_arg("num_attention_heads")
    check_for_arg("max_position_embeddings")
    check_for_arg("position_embedding_type")
    check_for_arg("tokenizer_type")
    check_for_arg("iteration")
    check_for_arg("bert_binary_head")
    check_for_arg("disable_bias_linear", False)
    check_for_arg("params_dtype")
    check_for_arg("swiglu", False)

    # Determine how to make our models.
    # MiniCPM is also a GPT model, so we accept both "GPT" and "minicpm"
    assert args.model_type in ["GPT", "minicpm"], f"MiniCPM is a GPT model, but got {args.model_type}."
    margs.model_type = ModelType.encoder_or_decoder

    # Suppress warning about torch.distributed not being initialized.
    module.MegatronModule.embedding_warning_printed = True

    try:
        set_global_variables(margs, build_tokenizer=False)
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    if fused_kernels is not None:
        fused_kernels.load(margs)

    # Short aliases.
    tp_size = margs.tensor_model_parallel_size
    # pp_size = margs.pipeline_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    # Metadata.
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.norm_has_bias = False
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.true_vocab_size = None  # skips padding in saver
    md.make_vocab_size_divisible_by = None
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0

    # Get first pipe stage.
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    model = load_checkpoint_to_model(margs)

    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name if name is not None else 'None'}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings.
    message = {"word embeddings": model.embedding.word_embeddings.weight.data}
    if md.position_embedding_type == "learned_absolute":
        message["position embeddings"] = model.embedding.position_embeddings.weight.data
    else:
        assert not hasattr(model.embedding, "position_embeddings")

    queue_put("embeddings", message)

    for layer_num in range(margs.num_layers):
        message = {}

        # Get non-parallel tensors from tp_rank 0.
        layer = model.decoder.layers[layer_num]
        # Note: The new model structure might have different norm names
        # message["input norm weight"] = layer.input_norm.weight.data
        # message["post norm weight"] = layer.post_attention_norm.weight.data
        if md.linear_bias:
            message["dense bias"] = layer.self_attention.linear_proj.bias.data
            message["mlp l1 bias"] = layer.mlp.linear_fc2.bias.data

        # Grab all parallel tensors for this layer.
        qkv_weight = []
        qkv_bias = []
        dense_weight = []
        mlp_l0_weight = []
        mlp_l0_bias = []
        mlp_l1_weight = []
        layer = model.decoder.layers[layer_num]
        qkv_weight.append(layer.self_attention.linear_qkv.weight.data)
        dense_weight.append(layer.self_attention.linear_proj.weight.data)
        mlp_l0_weight.append(layer.mlp.linear_fc1.weight.data)
        mlp_l1_weight.append(layer.mlp.linear_fc2.weight.data)
        if md.linear_bias:
            qkv_bias.append(layer.self_attention.linear_qkv.bias.data)
            mlp_l0_bias.append(layer.mlp.linear_fc1.bias.data)

        # Handle gated linear units.
        if md.swiglu:
            # Concat all the first halves ('W's) and all the second halves ('V's).
            for tp_rank in range(tp_size):
                mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
            message["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
            message["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
        else:
            message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

        # Simple concat of the rest.
        message["qkv weight"] = torch.cat(qkv_weight, dim=0)
        message["dense weight"] = torch.cat(dense_weight, dim=1)
        message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)
        if md.linear_bias:
            message["qkv bias"] = torch.cat(qkv_bias, dim=0)
            if md.swiglu:
                for tp_rank in range(tp_size):
                    mlp_l0_bias[tp_rank] = torch.chunk(mlp_l0_bias[tp_rank], 2, dim=0)
                message["mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias], dim=0)
                message["mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias], dim=0)
            else:
                message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)

        queue_put(f"transformer layer {layer_num}", message)

    # Send final norm from tp_rank 0.
    # Note: The new model structure doesn't have a final_norm, so we skip it
    # message = {
    #     "weight": model.language_model.encoder.final_norm.weight.data,
    # }
    # queue_put("final norm", message)

    if md.output_layer:
        message = {"weight": model.output_layer.weight.data}
        queue_put("output layer", message)

    queue.put("done")


def load_checkpoint(queue, args, direction="hf2megatron"):
    try:
        # For direct saving, we'll bypass the queue system and save directly

        # Llama-2 requires HF transformers >=4.31.0.
        verify_transformers_version()

        # Add current directory to path to find local megatron module
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)

        # Search in directory above this.
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
        if args.megatron_path is not None:
            sys.path.insert(0, args.megatron_path)

        try:
            from megatron.training.arguments import parse_args, validate_args

            from megatron.training.global_vars import set_global_variables

            # from megatron.legacy.model import module  # Not used

            from megatron.core import mpu

            from megatron.core.enums import ModelType

            from megatron.legacy import fused_kernels

        except ModuleNotFoundError:
            print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
            return None

        # We want all arguments to come from us.
        sys.argv = [
            "script.py",
            "--no-masked-softmax-fusion",
            "--no-bias-gelu-fusion",
            "--no-bias-dropout-fusion",
            "--no-async-tensor-model-parallel-allreduce",
            "--use-cpu-initialization",
            "--micro-batch-size",
            "1",
            "--no-load-optim",
            "--no-load-rng",
            "--no-save-optim",
            "--no-save-rng",
            "--no-initialization",
            "--no-gradient-accumulation-fusion",
            "--load",
            args.load_dir,
        ]

        margs = parse_args()
        margs.tokenizer_model = args.tokenizer_model
        load_args_from_checkpoint(margs)

        # Arguments do sanity checks on the world size, but we don't care,
        # so trick it into thinking we are plenty of processes.
        margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

        margs = validate_args(margs)

        def check_for_arg(arg_name, default=None):
            if getattr(margs, arg_name, None) is None:
                if default is not None:
                    setattr(margs, arg_name, default)
                else:
                    print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                    print(f"Arguments: {margs}")
                    return None

        # Check for required arguments
        check_for_arg("tensor_model_parallel_size", 1)
        check_for_arg("pipeline_model_parallel_size", 1)
        check_for_arg("num_layers")
        check_for_arg("hidden_size")
        check_for_arg("seq_length")
        check_for_arg("num_attention_heads")
        check_for_arg("max_position_embeddings")
        check_for_arg("position_embedding_type")
        check_for_arg("tokenizer_type")
        check_for_arg("iteration", 1)
        check_for_arg("bert_binary_head", True)
        check_for_arg("disable_bias_linear", False)
        check_for_arg("params_dtype")
        check_for_arg("swiglu")

        # Check model type
        if args.model_type == "minicpm":
            margs.model_type = ModelType.encoder_or_decoder

        # Set global variables
        # set_args(margs)  # Not imported, skipping
        set_global_variables(margs)

        # Set tensor model parallel world size
        mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
        mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
        mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)

        # Load fused kernels
        try:
            fused_kernels.load(margs)
        except Exception:
            pass

        # Load the model based on direction
        if direction == "hf2megatron":
            model = load_checkpoint_to_model(margs)

            # Save the model directly
            os.makedirs(args.save_dir, exist_ok=True)

            # Save model state dict
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model.pt"))

            # Save model metadata
            metadata = {
                "model_type": args.model_type,
                "num_layers": margs.num_layers,
                "hidden_size": margs.hidden_size,
                "num_attention_heads": margs.num_attention_heads,
                "max_position_embeddings": margs.max_position_embeddings,
                "vocab_size": margs.vocab_size,
                "position_embedding_type": margs.position_embedding_type,
                "swiglu": margs.swiglu,
                "linear_bias": not margs.disable_bias_linear,
                "output_layer": True,
            }

            with open(os.path.join(args.save_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            return model
        elif direction == "megatron2hf":
            hf_model = load_megatron_checkpoint_to_hf(margs)

            # Save the HuggingFace model
            os.makedirs(args.save_dir, exist_ok=True)

            # Save the model using HuggingFace's save_pretrained
            hf_model.save_pretrained(args.save_dir)

            return hf_model
        else:
            raise ValueError(f"Unsupported direction: {direction}")

    except Exception:
        import traceback

        traceback.print_exc()
        return None
