import pytest
import torch
import os
import shutil
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import write_basic_config # For default config if needed
from datasets import Dataset, DatasetDict
from safetensors.torch import save_file as save_safetensors

from sae_lens.config import LanguageModelSAERunnerConfig, CacheActivationsRunnerConfig, DTYPE_MAP
from sae_lens.sae_training_runner import SAETrainingRunner
from sae_lens.training.activations_store import ActivationsStore # To access store directly for some tests
from tests.training.test_fsdp_training import MODEL_TO_TRANSFORMER_LAYER_CLS # Reuse for FSDP config

# Default Accelerator config for tests if needed
if not os.path.exists(os.path.expanduser("~/.cache/huggingface/accelerate/default_config.yaml")):
    write_basic_config(mixed_precision="no") # Or "fp16", "bf16"

# Global accelerator for tests that need it at module level or for setup
# However, it's better to initialize it within each test function or fixture
# to ensure clean state, especially for distributed tests.

# Helper to create a dummy tokenized dataset
def create_dummy_tokenized_dataset(
    path: str, num_examples: int = 100, context_size: int = 64, vocab_size: int = 1000
):
    if os.path.exists(path):
        shutil.rmtree(path) # Clean up before creating
    os.makedirs(path, exist_ok=True)
    data = {
        "tokens": [
            torch.randint(0, vocab_size, (context_size,)).tolist()
            for _ in range(num_examples)
        ]
    }
    dataset = Dataset.from_dict(data)
    dataset.save_to_disk(path)
    return path

# Helper to create a dummy pre-computed activation dataset
def create_dummy_cached_activations(
    path: str, hook_name: str, num_examples: int = 100, context_size: int = 64, d_in: int = 32, dtype_str: str = "float32"
):
    if os.path.exists(path):
        shutil.rmtree(path) # Clean up
    os.makedirs(path, exist_ok=True)

    dtype = DTYPE_MAP[dtype_str]
    # Generate activations and corresponding token_ids
    activations_data = torch.randn(num_examples, context_size, d_in, dtype=dtype)
    token_ids_data = torch.randint(0, 1000, (num_examples, context_size), dtype=torch.long)

    data_dict = {
        hook_name: [activations_data[i] for i in range(num_examples)],
        "token_ids": [token_ids_data[i] for i in range(num_examples)],
    }

    dataset = Dataset.from_dict(data_dict)
    # Save in Hugging Face Dataset format
    dataset.save_to_disk(path)

    # Save a dummy config file as CacheActivationsRunner might expect it (though not strictly needed for store loading)
    # This is more for completeness if other parts of the code try to load it.
    dummy_cache_cfg = CacheActivationsRunnerConfig(
        dataset_path="dummy", # placeholder
        model_name="dummy_model",
        model_batch_size=num_examples,
        hook_name=hook_name,
        hook_layer=0, # placeholder
        d_in=d_in,
        training_tokens=num_examples * context_size,
        context_size=context_size,
        new_cached_activations_path=path,
        dtype=dtype_str,
    )
    # We don't have a to_json method in CacheActivationsRunnerConfig by default in the provided code
    # So, skip saving this config for now, or implement a simple save if needed.
    # with open(os.path.join(path, "sae_lens_config.json"), "w") as f:
    #     json.dump(dummy_cache_cfg.__dict__, f) # Basic dict dump

    return path, activations_data, token_ids_data


def distributed_test_config(
    model_name: str,
    dataset_path: str,
    cached_activations_path: str | None = None,
    use_cached_activations: bool = False,
    d_in_override: int | None = None, # For specific model hook d_in
    hook_name_override: str | None = None,
    fsdp_transformer_cls_override: list[str] | None = None,
) -> LanguageModelSAERunnerConfig:

    transformer_layer_cls = fsdp_transformer_cls_override or MODEL_TO_TRANSFORMER_LAYER_CLS.get(model_name, [])

    # Determine d_in based on model if not overridden
    if d_in_override:
        actual_d_in = d_in_override
    elif model_name == "EleutherAI/pythia-14m":
        actual_d_in = 1024 # MLP out for pythia-14m (256 model dim * 4)
    elif model_name == "gpt2":
        actual_d_in = 3072 # MLP out for gpt2 (768 model dim * 4)
    else: # Default or raise error
        # This should be set based on the model's actual architecture
        raise ValueError(f"Please provide d_in_override or add specific d_in for {model_name}")

    actual_hook_name = hook_name_override
    if not actual_hook_name:
        if model_name == "EleutherAI/pythia-14m":
            actual_hook_name = "gpt_neox.layers.0.mlp.dense_4h_to_h"
        elif model_name == "gpt2":
            actual_hook_name = "blocks.0.hook_mlp_out"
        else:
            raise ValueError(f"Please provide hook_name_override or add specific hook_name for {model_name}")

    return LanguageModelSAERunnerConfig(
        model_name=model_name,
        model_class_name="HookedTransformer",
        hook_name=actual_hook_name,
        hook_layer=0,
        dataset_path=dataset_path,
        streaming=not use_cached_activations, # Stream if not cached, not relevant if cached
        context_size=64,

        use_cached_activations=use_cached_activations,
        cached_activations_path=cached_activations_path,

        d_in=actual_d_in,
        d_sae=actual_d_in * 2, # Small expansion

        training_tokens= (64 * 10), # Very few tokens: context_size * num_unique_prompts_to_process
        store_batch_size_prompts=2, # Global batch size for LLM processing
        train_batch_size_tokens=128, # For SAE training
        n_batches_in_buffer=5, # store_batch_size_prompts * n_batches_in_buffer <= training_tokens / context_size
                                # 2 * 5 = 10 prompts total in buffer. training_tokens implies 10 prompts.

        fsdp_enabled=True,
        fsdp_sharding_strategy="FULL_SHARD",
        fsdp_auto_wrap_policy="transformer_auto_wrap_policy",
        fsdp_transformer_layer_cls_to_wrap=transformer_layer_cls,
        fsdp_offload_params=False,
        fsdp_use_orig_params=True,

        device="cuda", # Accelerator handles actual device assignment
        log_to_wandb=False,
        n_checkpoints=0,
        checkpoint_path=f"/tmp/test_dist_act_store_checkpoints_{os.getpid()}",
        dtype="float32",
        autocast=False, # Keep False for FSDP debugging simplicity
        act_store_device="cpu", # Buffer on CPU
    )

# Note: These tests need to be run with `accelerate launch pytest tests/training/test_distributed_activations_store.py`

@pytest.mark.distributed_tests # Custom marker
def test_on_the_fly_generation_distributed():
    """
    Test on-the-fly activation generation with an FSDP-wrapped LLM.
    Verifies that the ActivationsStore gathers activations correctly and the buffer is consistent.
    """
    accelerator = Accelerator()
    if accelerator.num_processes == 1:
        pytest.skip("This test requires a distributed environment (num_processes > 1). Run with `accelerate launch`.")
    if not torch.cuda.is_available():
        pytest.skip("Distributed test requires CUDA.")

    model_key = "EleutherAI/pythia-14m" # Small model for faster test
    # Using a readily available dataset for now. For robust CI, generate a dummy one.
    dataset_p = "stas/c4-en-10k"

    # Temp dir for any checkpointing by runner, though n_checkpoints=0
    test_checkpoint_dir = f"/tmp/test_on_the_fly_dist_checkpoints_{os.getpid()}"

    cfg = distributed_test_config(
        model_name=model_key,
        dataset_path=dataset_p,
        d_in_override=1024, # pythia-14m mlp out
        hook_name_override="gpt_neox.layers.0.mlp.dense_4h_to_h",
        fsdp_transformer_cls_override=["GPTNeoXLayer"]
    )
    cfg.checkpoint_path = test_checkpoint_dir
    cfg.training_tokens = cfg.store_batch_size_prompts * cfg.context_size * cfg.n_batches_in_buffer # Ensure buffer can be filled once

    try:
        runner = SAETrainingRunner(cfg=cfg) # Initializes LLM (FSDP wrapped) and ActivationsStore

        # Fill the buffer once. get_buffer is called internally by next_batch -> get_data_loader -> get_buffer
        # We want to inspect the state of the _storage_buffer after it's filled.
        # The _storage_buffer should be half of n_batches_in_buffer * store_batch_size_prompts * context_size tokens
        # Let's call get_buffer directly to control the size for testing.

        # The buffer size is store_batch_size_prompts * n_batches_in_buffer * context_size tokens
        # Example: 2 prompts/batch * 5 batches in buffer * 64 ctx = 640 tokens total in a full buffer load
        # ActivationsStore.storage_buffer holds half of this after first fill.
        # So, 320 tokens worth of activations.

        buffer_activations_tuple = runner.activations_store.get_buffer(runner.activations_store.n_batches_in_buffer)
        # buffer_activations_tuple is (activations_tensor, tokens_tensor_or_None)
        # activations_tensor shape: (total_tokens_in_buffer, num_layers=1, d_in)

        accelerator.wait_for_everyone() # Ensure all processes have processed get_buffer

        # Verify buffer content consistency across ranks
        # Gather the shape of the buffer from all ranks
        buffer_shape_tensor = torch.tensor(buffer_activations_tuple[0].shape, device=accelerator.device)
        gathered_shapes = accelerator.gather(buffer_shape_tensor)

        if accelerator.is_main_process:
            # Check all gathered shapes are identical
            first_shape = gathered_shapes[0]
            assert all(torch.equal(s, first_shape) for s in gathered_shapes), "Buffer shapes differ across ranks"

            # Expected shape for activations: (global_prompts_in_buffer * context_size, 1, d_in)
            # global_prompts_in_buffer = cfg.store_batch_size_prompts * cfg.n_batches_in_buffer
            # Here, buffer_activations_tuple[0] is the direct output of get_buffer, so it's the full buffer load.
            expected_num_tokens = cfg.store_batch_size_prompts * cfg.n_batches_in_buffer * cfg.context_size
            assert buffer_activations_tuple[0].shape[0] == expected_num_tokens, \
                f"Buffer activation count {buffer_activations_tuple[0].shape[0]} does not match expected {expected_num_tokens}"
            assert buffer_activations_tuple[0].shape[2] == cfg.d_in

        # To check content, gather a hash or a small part of the tensor
        # This assumes buffer_activations_tuple[0] is on CPU (cfg.act_store_device="cpu")
        # If it's large, hashing a slice might be better.
        # For simplicity, let's assume it's on CPU and not excessively large for this test config.

        # Sum of first activation vector as a simple content check
        sum_on_this_rank = buffer_activations_tuple[0].sum().to(accelerator.device) # Move to GPU for gather
        gathered_sums = accelerator.gather(sum_on_this_rank)

        if accelerator.is_main_process:
            first_sum = gathered_sums[0]
            assert torch.allclose(gathered_sums, first_sum, atol=1e-5, rtol=1e-3), \
                f"Buffer content (sums) differ across ranks. Sums: {gathered_sums}"

    finally:
        if os.path.exists(test_checkpoint_dir):
            shutil.rmtree(test_checkpoint_dir)


@pytest.mark.distributed_tests
def test_precomputed_loading_distributed():
    accelerator = Accelerator()
    if accelerator.num_processes == 1:
        pytest.skip("This test requires a distributed environment (num_processes > 1). Run with `accelerate launch`.")
    if not torch.cuda.is_available(): # Though LLM isn't used, Accelerator might default to CUDA for gather ops
        pytest.skip("Distributed test might require CUDA for accelerator ops.")

    model_key = "EleutherAI/pythia-14m" # Used for config, but LLM won't be run
    hook_name_for_cache = "blocks.0.hook_mlp_out" # Arbitrary for cache structure
    d_in_for_cache = 32 # Small d_in for dummy data
    context_size_for_cache = 64
    num_cached_examples = cfg.store_batch_size_prompts * cfg.n_batches_in_buffer * 2 # Enough for a couple of buffer fills
                                                                                 # Example: 2 * 5 * 2 = 20 prompts

    dummy_dataset_path = Path(f"/tmp/dummy_dist_text_dataset_{os.getpid()}")
    dummy_cached_act_path = Path(f"/tmp/dummy_dist_cached_act_{os.getpid()}")

    try:
        # Create dummy tokenized dataset (not strictly needed if only using cached activations, but good for full config)
        create_dummy_tokenized_dataset(str(dummy_dataset_path), num_examples=num_cached_examples, context_size=context_size_for_cache)

        # Create dummy cached activations
        _, ref_acts_data, ref_tokens_data = create_dummy_cached_activations(
            str(dummy_cached_act_path),
            hook_name=hook_name_for_cache,
            num_examples=num_cached_examples,
            context_size=context_size_for_cache,
            d_in=d_in_for_cache
        )

        cfg = distributed_test_config(
            model_name=model_key, # Not loaded, but influences config like d_sae
            dataset_path=str(dummy_dataset_path),
            cached_activations_path=str(dummy_cached_act_path),
            use_cached_activations=True,
            d_in_override=d_in_for_cache, # d_in must match cached data
            hook_name_override=hook_name_for_cache, # hook_name must match
            fsdp_transformer_cls_override=["GPTNeoXLayer"] # For FSDP init, though model not run
        )
        # Ensure enough tokens are requested to fill the buffer once from cache
        cfg.training_tokens = cfg.store_batch_size_prompts * cfg.n_batches_in_buffer * cfg.context_size
        cfg.checkpoint_path = f"/tmp/test_precomputed_dist_checkpoints_{os.getpid()}"


        runner = SAETrainingRunner(cfg=cfg) # Initializes ActivationsStore with cached path

        # Get one full buffer's worth of data
        # store_batch_size_prompts * n_batches_in_buffer examples will be loaded globally
        # Each example has context_size tokens. Activations are (tokens, 1, d_in)
        buffer_activations_tuple = runner.activations_store.get_buffer(runner.activations_store.n_batches_in_buffer)

        accelerator.wait_for_everyone()

        # Verify buffer content and consistency
        buffer_acts_content = buffer_activations_tuple[0].cpu() # (N, 1, d_in)
        buffer_tokens_content = buffer_activations_tuple[1].cpu() # (N) (tokens are flattened)

        # Reshape reference data to match buffer format (flattened tokens)
        # ref_acts_data is (num_cached_examples, context_size, d_in)
        # ref_tokens_data is (num_cached_examples, context_size)
        # Buffer loads `cfg.store_batch_size_prompts * cfg.n_batches_in_buffer` examples.
        num_examples_in_buffer = cfg.store_batch_size_prompts * cfg.n_batches_in_buffer

        expected_acts_flat = ref_acts_data[:num_examples_in_buffer].reshape(-1, 1, d_in_for_cache)
        expected_tokens_flat = ref_tokens_data[:num_examples_in_buffer].reshape(-1)

        # Gather shapes and content sum for cross-rank consistency check
        shape_tensor = torch.tensor(buffer_acts_content.shape, device=accelerator.device)
        sum_tensor = buffer_acts_content.sum().to(accelerator.device)

        gathered_shapes = accelerator.gather(shape_tensor)
        gathered_sums = accelerator.gather(sum_tensor)

        if accelerator.is_main_process:
            first_shape = gathered_shapes[0]
            assert all(torch.equal(s, first_shape) for s in gathered_shapes), "Buffer shapes differ across ranks for cached data"
            assert buffer_acts_content.shape == expected_acts_flat.shape, \
                f"Buffer acts shape {buffer_acts_content.shape} mismatch with expected {expected_acts_flat.shape}"
            if buffer_tokens_content is not None and expected_tokens_flat is not None:
                 assert buffer_tokens_content.shape == expected_tokens_flat.shape, \
                    f"Buffer tokens shape {buffer_tokens_content.shape} mismatch with expected {expected_tokens_flat.shape}"

            first_sum = gathered_sums[0]
            assert torch.allclose(gathered_sums, first_sum, atol=1e-5, rtol=1e-3), \
                f"Buffer content (sums) differ across ranks for cached data. Sums: {gathered_sums}"

            # Content verification (on main process, as all should be identical)
            assert torch.allclose(buffer_acts_content, expected_acts_flat, atol=1e-5), "Cached activation content mismatch"
            if buffer_tokens_content is not None and expected_tokens_flat is not None:
                assert torch.equal(buffer_tokens_content, expected_tokens_flat), "Cached token content mismatch"

    finally:
        if accelerator.is_main_process: # Only main process should delete shared resources
            if os.path.exists(dummy_dataset_path):
                shutil.rmtree(dummy_dataset_path)
            if os.path.exists(dummy_cached_act_path):
                shutil.rmtree(dummy_cached_act_path)
            if os.path.exists(cfg.checkpoint_path):
                shutil.rmtree(cfg.checkpoint_path)

def test_placeholder_to_ensure_file_is_valid_module(): # To satisfy pytest if other tests are skipped
    assert True

```
