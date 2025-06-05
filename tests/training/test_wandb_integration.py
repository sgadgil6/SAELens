import pytest
import torch
import os
import shutil
from unittest.mock import patch, MagicMock

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae_training_runner import SAETrainingRunner
from accelerate import Accelerator

# Helper for minimal config
def wandb_test_config(
    checkpoint_path_root="/tmp",
    use_cached_acts=False,
    cached_acts_path=None,
    dataset_path="stas/c4-en-10k", # A small, accessible dataset
    model_name="EleutherAI/pythia-14m" # A small, accessible model
    ) -> LanguageModelSAERunnerConfig:

    # Construct a unique checkpoint path for each test run to avoid conflicts
    run_specific_checkpoint_path = os.path.join(checkpoint_path_root, f"wandb_test_checkpoints_{os.getpid()}_{torch.randint(0, 10000, (1,)).item()}")

    return LanguageModelSAERunnerConfig(
        # Model & Data
        model_name=model_name,
        model_class_name="HookedTransformer",
        hook_name="gpt_neox.layers.0.mlp.dense_4h_to_h", # For pythia-14m
        hook_layer=0,
        dataset_path=dataset_path,
        streaming=not use_cached_acts,
        context_size=64,
        use_cached_activations=use_cached_acts,
        cached_activations_path=cached_acts_path,

        # SAE Params
        d_in=1024, # pythia-14m mlp out
        d_sae=2048,

        # Training Params
        training_tokens=64 * 10, # Very few: context_size * num_prompts
        store_batch_size_prompts=2,
        train_batch_size_tokens=128,
        n_batches_in_buffer=5,

        # WandB
        log_to_wandb=True,
        wandb_project="sae-lens-test-project",
        wandb_entity="test-entity", # Replace with your test entity or use a mock
        wandb_id=f"test_wandb_id_{os.getpid()}_{torch.randint(0, 10000, (1,)).item()}",
        run_name="test_wandb_run",

        # Other
        device="cpu", # Test on CPU to avoid GPU requirements for this specific test
        act_store_device="cpu",
        seed=42,
        n_checkpoints=0,
        checkpoint_path=run_specific_checkpoint_path,
        dtype="float32",
        autocast=False,
        # Disable FSDP for this test to focus on wandb
        fsdp_enabled=False,
    )

# Environment variable to control running of these tests
# Set SAE_LENS_RUN_WANDB_TESTS="true" to run them
# These tests might make actual calls to WandB if not fully mocked or if wandb is in online mode.
# For CI, ensure wandb is configured to be offline or that API keys are handled securely if online tests are intended.
# A simple way to ensure offline for tests is to set WANDB_MODE=offline environment variable.
RUN_WANDB_TESTS = os.environ.get("SAE_LENS_RUN_WANDB_TESTS", "false").lower() == "true"
REASON_WANDB_TESTS_SKIPPED = "WandB integration tests are skipped. Set SAE_LENS_RUN_WANDB_TESTS=true to run."

@pytest.mark.wandb_integration
@pytest.mark.skipif(not RUN_WANDB_TESTS, reason=REASON_WANDB_TESTS_SKIPPED)
@patch.object(Accelerator, 'end_training')
@patch.object(Accelerator, 'log')
@patch.object(Accelerator, 'init_trackers')
def test_wandb_tracking_via_accelerate(
    mock_init_trackers: MagicMock,
    mock_log: MagicMock,
    mock_end_training: MagicMock,
):
    """
    Tests that SAETrainingRunner correctly uses Accelerator for WandB tracking.
    """
    cfg = wandb_test_config()

    # Ensure wandb is in offline mode for this test to prevent actual cloud logging
    original_wandb_mode = os.environ.get("WANDB_MODE")
    os.environ["WANDB_MODE"] = "offline"

    runner = None
    try:
        # We are not creating a real model or real data for this test,
        # as SAETrainingRunner.__init__ and .run() will try to load them.
        # The goal is to check if the accelerator's tracking methods are called.
        # This means we need to mock parts of SAETrainingRunner or make it runnable
        # with minimal external dependencies.
        # For now, let's assume the config is enough and see where it fails if model/data loading is strict.
        # The current SAETrainingRunner loads model and data in __init__.
        # This test will be more of an integration test of the runner's wandb logic.

        # To make this test runnable without real model/data,
        # we would typically mock `load_model` and parts of `ActivationsStore`.
        # However, the task is to test the WandB integration part of the runner.
        # Let's allow it to proceed as much as possible and see if trackers are called.
        # The test uses a small, real model and dataset to ensure the runner initializes.

        runner = SAETrainingRunner(cfg=cfg)
        runner.run()

        # Assertions
        mock_init_trackers.assert_called_once()
        args, kwargs = mock_init_trackers.call_args
        assert args[0] == cfg.wandb_project # project_name
        assert kwargs['config'] == cfg.to_dict()
        assert "wandb" in kwargs['init_kwargs']
        assert kwargs['init_kwargs']['wandb']['entity'] == cfg.wandb_entity
        assert kwargs['init_kwargs']['wandb']['id'] == cfg.wandb_id
        assert kwargs['init_kwargs']['wandb']['name'] == cfg.run_name

        # Check if log was called. The number of calls can be >0.
        # Exact number depends on log_frequency and how many steps run.
        # For this minimal run (640 training tokens, batch size 128 -> 5 steps),
        # and wandb_log_frequency=10, it might be called for sparsity init + 0-1 training steps.
        # Let's just check it was called.
        assert mock_log.called, "Accelerator.log was not called."

        mock_end_training.assert_called_once()

    finally:
        if original_wandb_mode is None:
            del os.environ["WANDB_MODE"]
        else:
            os.environ["WANDB_MODE"] = original_wandb_mode

        # Clean up checkpoint directory
        if os.path.exists(cfg.checkpoint_path):
            shutil.rmtree(cfg.checkpoint_path)

def test_placeholder_to_ensure_file_is_valid_module():
    assert True

```
