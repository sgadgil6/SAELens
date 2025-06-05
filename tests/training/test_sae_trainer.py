from pathlib import Path
from typing import Any, Callable

import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file
from transformer_lens import HookedTransformer

from sae_lens import __version__
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae_training_runner import SAETrainingRunner
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.sae_trainer import (
    SAETrainer,
    TrainStepOutput,
    _log_feature_sparsity,
    _update_sae_lens_training_version,
)
from sae_lens.training.training_sae import TrainingSAE
from tests.helpers import TINYSTORIES_MODEL, build_sae_cfg, load_model_cached


@pytest.fixture
def cfg():
    return build_sae_cfg(d_in=64, d_sae=128, hook_layer=0)


@pytest.fixture
def model():
    return load_model_cached(TINYSTORIES_MODEL)


@pytest.fixture
def activation_store(model: HookedTransformer, cfg: LanguageModelSAERunnerConfig):
    return ActivationsStore.from_config(
        model, cfg, override_dataset=Dataset.from_list([{"text": "hello world"}] * 2000)
    )


@pytest.fixture
def training_sae(cfg: LanguageModelSAERunnerConfig):
    return TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())


@pytest.fixture
def trainer(
    cfg: LanguageModelSAERunnerConfig,
    training_sae: TrainingSAE,
    model: HookedTransformer,
    activation_store: ActivationsStore,
):
    return SAETrainer(
        model=model,
        sae=training_sae,
        activation_store=activation_store,
        save_checkpoint_fn=lambda *args, **kwargs: None,  # noqa: ARG005
        cfg=cfg,
    )


def modify_sae_output(sae: TrainingSAE, modifier: Callable[[torch.Tensor], Any]):
    """
    Helper to modify the output of the SAE forward pass for use in patching, for use in patch side_effect.
    We need real grads during training, so we can't just mock the whole forward pass directly.
    """

    def modified_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        output = TrainingSAE.forward(sae, *args, **kwargs)
        return modifier(output)

    return modified_forward


def test_train_step__reduces_loss_when_called_repeatedly_on_same_acts(
    trainer: SAETrainer,
) -> None:
    layer_acts = trainer.activations_store.next_batch()

    # intentionally train on the same activations 5 times to ensure loss decreases
    train_outputs = [
        trainer._train_step(
            sae=trainer.sae,
            sae_in=layer_acts[:, 0, :],
        )
        for _ in range(5)
    ]

    # ensure loss decreases with each training step
    for output, next_output in zip(train_outputs[:-1], train_outputs[1:]):
        assert output.loss > next_output.loss
    assert (
        trainer.n_frac_active_tokens == 20
    )  # should increment each step by batch_size (5*4)


def test_train_step__output_looks_reasonable(trainer: SAETrainer) -> None:
    layer_acts = trainer.activations_store.next_batch()

    output = trainer._train_step(
        sae=trainer.sae,
        sae_in=layer_acts[:, 0, :],
    )

    assert output.loss > 0
    # only hook_point_layer=0 acts should be passed to the SAE
    assert torch.allclose(output.sae_in, layer_acts[:, 0, :])
    assert output.sae_out.shape == output.sae_in.shape
    assert output.feature_acts.shape == (4, 128)  # batch_size, d_sae
    # ghots grads shouldn't trigger until dead_feature_window, which hasn't been reached yet
    assert output.losses.get("ghost_grad_loss", 0) == 0
    assert trainer.n_frac_active_tokens == 4
    assert trainer.act_freq_scores.sum() > 0  # at least SOME acts should have fired
    assert torch.allclose(
        trainer.act_freq_scores, (output.feature_acts.abs() > 0).float().sum(0)
    )


def test_train_step__sparsity_updates_based_on_feature_act_sparsity(
    trainer: SAETrainer,
) -> None:
    trainer._reset_running_sparsity_stats()
    layer_acts = trainer.activations_store.next_batch()

    train_output = trainer._train_step(
        sae=trainer.sae,
        sae_in=layer_acts[:, 0, :],
    )
    feature_acts = train_output.feature_acts

    # should increase by batch_size
    assert trainer.n_frac_active_tokens == 4
    # add freq scores for all non-zero feature acts
    assert torch.allclose(
        trainer.act_freq_scores, (feature_acts > 0).float().sum(dim=0)
    )

    # check that features that just fired have n_forward_passes_since_fired = 0
    assert (
        trainer.n_forward_passes_since_fired[
            ((feature_acts > 0).float()[-1] == 1)
        ].max()
        == 0
    )
    assert train_output.feature_acts is feature_acts


def test_log_feature_sparsity__handles_zeroes_by_default_fp32() -> None:
    fp32_zeroes = torch.tensor([0], dtype=torch.float32)
    assert _log_feature_sparsity(fp32_zeroes).item() != float("-inf")


# TODO: currently doesn't work for fp16, we should address this
@pytest.mark.skip(reason="Currently doesn't work for fp16")
def test_log_feature_sparsity__handles_zeroes_by_default_fp16() -> None:
    fp16_zeroes = torch.tensor([0], dtype=torch.float16)
    assert _log_feature_sparsity(fp16_zeroes).item() != float("-inf")


def test_build_train_step_log_dict(trainer: SAETrainer) -> None:
    train_output = TrainStepOutput(
        sae_in=torch.tensor([[-1, 0], [0, 2], [1, 1]]).float(),
        sae_out=torch.tensor([[0, 0], [0, 2], [0.5, 1]]).float(),
        feature_acts=torch.tensor([[0, 0, 0, 1], [1, 0, 0, 1], [1, 0, 1, 1]]).float(),
        hidden_pre=torch.tensor([[-1, 0, 0, 1], [1, -1, 0, 1], [1, -1, 1, 1]]).float(),
        loss=torch.tensor(0.5),
        losses={
            "mse_loss": 0.25,
            "l1_loss": 0.1,
            "ghost_grad_loss": 0.15,
        },
    )

    # we're relying on the trainer only for some of the metrics here
    # we should more / less try to break this and push
    # everything through the train step output if we can.
    log_dict = trainer._build_train_step_log_dict(
        output=train_output, n_training_tokens=123
    )
    assert log_dict == {
        "losses/mse_loss": 0.25,
        # l1 loss is scaled by l1_coefficient
        "losses/l1_loss": train_output.losses["l1_loss"] / trainer.cfg.l1_coefficient,
        "losses/raw_l1_loss": train_output.losses["l1_loss"],
        "losses/overall_loss": 0.5,
        "losses/ghost_grad_loss": 0.15,
        "metrics/explained_variance": 0.6875,
        "metrics/explained_variance_legacy": 0.75,
        "metrics/explained_variance_legacy_std": 0.25,
        "metrics/l0": 2.0,
        "sparsity/mean_passes_since_fired": trainer.n_forward_passes_since_fired.mean().item(),
        "sparsity/dead_features": trainer.dead_neurons.sum().item(),
        "details/current_learning_rate": 2e-4,
        "details/current_l1_coefficient": trainer.cfg.l1_coefficient,
        "details/n_training_tokens": 123,
    }


def test_train_sae_group_on_language_model__runs(
    ts_model: HookedTransformer,
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    cfg = build_sae_cfg(
        checkpoint_path=str(checkpoint_dir),
        training_tokens=20,
        context_size=8,
    )
    # just a tiny datast which will run quickly
    dataset = Dataset.from_list([{"text": "hello world"}] * 100)
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())
    sae = SAETrainer(
        model=ts_model,
        sae=sae,
        activation_store=activation_store,
        save_checkpoint_fn=lambda *args, **kwargs: None,  # noqa: ARG005
        cfg=cfg,
    ).fit()

    assert isinstance(sae, TrainingSAE)


def test_update_sae_lens_training_version_sets_the_current_version():
    cfg = build_sae_cfg(sae_lens_training_version="0.1.0")
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())
    _update_sae_lens_training_version(sae)
    assert sae.cfg.sae_lens_training_version == str(__version__)


def test_estimated_norm_scaling_factor_persistence(
    ts_model: HookedTransformer,
    tmp_path: Path,
):
    """Test that estimated_norm_scaling_factor is correctly persisted in intermediate checkpoints
    but not in the final checkpoint."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    cfg = build_sae_cfg(
        checkpoint_path=str(checkpoint_dir),
        training_tokens=100,  # Increased to ensure we hit checkpoints
        context_size=8,
        normalize_activations="expected_average_only_in",
        n_checkpoints=2,  # Explicitly request 2 checkpoints during training
    )

    # Create a small dataset
    dataset = Dataset.from_list([{"text": "hello world"}] * 100)
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    trainer = SAETrainer(
        model=ts_model,
        sae=sae,
        activation_store=activation_store,
        save_checkpoint_fn=SAETrainingRunner.save_checkpoint,
        cfg=cfg,
    )

    # Train the model - this should create checkpoints
    trainer.fit()
    checkpoint_paths = list(
        checkpoint_dir.glob("**/activations_store_state.safetensors")
    )
    # We should have exactly 2 checkpoints:
    assert (
        len(checkpoint_paths) == 2
    ), f"Expected 2 checkpoints but got {len(checkpoint_paths)}"
    during_checkpoints = [
        load_file(path) for path in checkpoint_paths if "final" not in path.parent.name
    ]
    final_checkpoints = [
        load_file(path) for path in checkpoint_paths if "final" in path.parent.name
    ]
    assert (
        len(during_checkpoints) == 1
    ), f"Expected 1 other checkpoint but got {len(during_checkpoints)}"
    assert (
        len(final_checkpoints) == 1
    ), f"Expected 1 final checkpoint but got {len(final_checkpoints)}"
    during_checkpoint = during_checkpoints[0]
    final_checkpoint = final_checkpoints[0]

    # Check intermediate checkpoints have the scaling factor
    assert "estimated_norm_scaling_factor" in during_checkpoint
    assert during_checkpoint["estimated_norm_scaling_factor"] is not None

    # Final checkpoint should NOT have the scaling factor as it's been folded into the weights
    assert "estimated_norm_scaling_factor" not in final_checkpoint


def test_layer_acts_device_handling_with_accelerator(
    cfg: LanguageModelSAERunnerConfig, # Use existing fixture, override device if needed
    model: HookedTransformer, # Mocked or real, SAETrainer doesn't use it much in _train_step
):
    from unittest.mock import MagicMock, PropertyMock
    from accelerate import Accelerator

    # 1. Setup Accelerator
    # For this test, a CPU accelerator is fine to check logic,
    # but testing with CUDA would be more thorough if environment allows.
    # If CUDA is available, use it, otherwise CPU.
    if torch.cuda.is_available():
        accelerator = Accelerator(device_placement=True) # Let accelerator place on its device (GPU)
        test_device = accelerator.device
    else:
        accelerator = Accelerator(device_placement=True, cpu=True) # Force CPU if no CUDA
        test_device = torch.device("cpu")

    cfg.device = str(test_device) # Ensure config reflects accelerator's device for SAE model placement
    cfg.act_store_device = "cpu" # Simulate activations coming from CPU buffer

    # 2. Mock ActivationsStore
    mock_activation_store = MagicMock(spec=ActivationsStore)
    # Simulate next_batch() returning a tensor on CPU
    cpu_tensor = torch.randn(cfg.train_batch_size_tokens // cfg.context_size, 1, cfg.d_in).to("cpu")
    # The actual next_batch() in code is `self.activations_store.next_batch()[:, 0, :]`
    # So the mock should return a tensor that can be sliced like that.
    # Let's make it (batch_prompts, num_layers_always_1, d_in) -> after slicing -> (batch_prompts, d_in)
    # The slicing in SAETrainer is `[:, 0, :]` on the output of `next_batch()`.
    # `next_batch` itself yields batches of shape (prompts, layers, d_in).
    # So, if train_batch_size_tokens = 4096, context_size=64, prompts = 4096/64 = 64.
    # The mock should return (64, 1, d_in)
    num_prompts_in_sae_batch = cfg.train_batch_size_tokens // cfg.context_size
    mock_batch_on_cpu = torch.randn(num_prompts_in_sae_batch, 1, cfg.d_in).to("cpu")
    mock_activation_store.next_batch.return_value = mock_batch_on_cpu

    # Mock other necessary attributes/methods if SAETrainer's __init__ or fit calls them
    # For _train_step, only next_batch and some config attributes might be needed from store.
    # The store itself is prepared, so it needs to be an object that can be prepared.
    # Making it a MagicMock might be too simplistic if `prepare` tries to access many attributes.
    # Let's use a real ActivationsStore but ensure its .next_batch() is patched.
    # This is safer.

    # Create a real store, then patch its next_batch
    # Need a dataset for the real store.
    dummy_dataset = Dataset.from_list([{"text": "example"}] * (num_prompts_in_sae_batch * 2)) # Enough for one batch

    # We need to ensure the store's internal device for its buffers is CPU.
    # The cfg.act_store_device="cpu" handles this.
    # The model passed to ActivationsStore is the LLM, not the SAE.
    # For this test, the LLM isn't used by the part of ActivationsStore we care about for yielding batches to SAE.
    # So, a simple mock LLM is fine.
    mock_llm = MagicMock(spec=HookedTransformer)
    # If tokenizer is accessed:
    mock_llm.tokenizer = MagicMock()
    mock_llm.tokenizer.bos_token_id = 0


    # Use a real ActivationStore, but we will mock its `next_batch` method after it's prepared.
    # The key is that the *prepared* store's `next_batch` output is what we care about.
    # However, Accelerator wraps the object. Mocking after prepare is tricky.
    # Alternative: Mock the class, then when it's instantiated, replace its next_batch.

    # Let's try mocking the instance's method *before* prepare, then check if prepare preserves the mock
    # or if we need to mock on the wrapped object.
    # This usually doesn't work as prepare creates a new wrapper.

    # The simplest for this test:
    # The ActivationsStore itself is prepared. When its `next_batch` (which is a method on the class)
    # is called on the *prepared* object, Accelerator should move its output.
    # So, the mock should be on the original object's `next_batch` to return CPU tensor.

    # Create a real store. Its `next_batch` will internally use a DataLoader on CPU tensors.
    # This is actually what we want to test: does Accelerator move the CPU tensor from the real store's DataLoader?
    # So, no need to mock next_batch to return a CPU tensor, it already should.

    # The cfg.act_store_device = "cpu" ensures the internal DataLoader of ActivationsStore yields CPU tensors.
    # This is the most realistic setup for the test.
    activation_store_instance = ActivationsStore.from_config(
        mock_llm, # Mocked LLM
        cfg,
        override_dataset=dummy_dataset,
        accelerator=accelerator # Pass accelerator for its device info if needed by store internally
    )


    # 3. Mock TrainingSAE
    mock_sae_model = MagicMock(spec=TrainingSAE)
    # `training_forward_pass` needs to return a TrainStepOutput object.
    # We need to capture the device of `sae_in`.
    sae_input_device_capture = []
    def capture_sae_in_device_forward(*args, sae_in: torch.Tensor, **kwargs):
        sae_input_device_capture.append(sae_in.device)
        # Return a dummy TrainStepOutput
        return TrainStepOutput(
            sae_in=sae_in, # Pass through to check if it's modified
            sae_out=torch.zeros_like(sae_in),
            feature_acts=torch.zeros((sae_in.shape[0], cfg.d_sae), device=sae_in.device),
            loss=torch.tensor(0.0, device=sae_in.device),
            mse_loss=torch.tensor(0.0, device=sae_in.device),
            l1_loss=torch.tensor(0.0, device=sae_in.device),
            ghost_grad_loss=torch.tensor(0.0, device=sae_in.device),
            losses={
                "mse_loss": torch.tensor(0.0, device=sae_in.device),
                "l1_loss": torch.tensor(0.0, device=sae_in.device),
                "ghost_grad_loss": torch.tensor(0.0, device=sae_in.device),
            }
        )
    mock_sae_model.training_forward_pass.side_effect = capture_sae_in_device_forward
    # Make sure the mock SAE has a .cfg attribute if trainer tries to access it
    mock_sae_model.cfg = MagicMock()
    mock_sae_model.cfg.normalize_sae_decoder = False # Or True, doesn't matter much for this test
    mock_sae_model.cfg.device = str(test_device) # So that sae.to(device) inside trainer doesn't complain if called
                                                 # (though prepare should handle device)
    # If parameters are accessed by optimizer
    mock_sae_model.parameters.return_value = [torch.nn.Parameter(torch.randn(10, device=test_device))]


    # 4. Instantiate SAETrainer
    # The SAETrainer will prepare the activation_store_instance and mock_sae_model
    trainer = SAETrainer(
        model=MagicMock(spec=HookedTransformer), # Mocked main model, not used in _train_step directly
        sae=mock_sae_model,
        activation_store=activation_store_instance,
        save_checkpoint_fn=lambda *args, **kwargs: None,
        cfg=cfg,
        accelerator=accelerator,
    )

    # 5. Call _train_step (or simplified loop)
    # The `fit` loop calls `activations_store.next_batch()[:, 0, :]`
    # Then passes this to `_train_step` as `sae_in`.
    # The key is that `trainer.activations_store` is the *prepared* version.

    # Simulate one step of the loop in `fit()`:
    # This ensures we call next_batch() on the *prepared* activation_store.
    # The output of ActivationsStore.next_batch() is (batch_prompts, num_layers, d_in).
    # It's then sliced to (batch_prompts, d_in) for the SAE.
    layer_acts_from_prepared_store = trainer.activations_store.next_batch()
    sliced_layer_acts = layer_acts_from_prepared_store[:, 0, :]

    trainer._train_step(
        sae=trainer.sae, # This is the prepared SAE
        sae_in=sliced_layer_acts # This is the tensor whose device we want to check *inside* training_forward_pass
    )

    # 6. Assert device
    assert len(sae_input_device_capture) == 1, "training_forward_pass was not called once."
    # The device of sae_in *inside* the forward pass of the *prepared* sae model
    # should be the accelerator's device.
    assert sae_input_device_capture[0] == accelerator.device, \
        f"Device of sae_in was {sae_input_device_capture[0]}, expected {accelerator.device}"

    # Also, the original layer_acts_from_prepared_store should be on accelerator.device
    assert layer_acts_from_prepared_store.device == accelerator.device, \
        f"Output of prepared_activations_store.next_batch() was on {layer_acts_from_prepared_store.device}, expected {accelerator.device}"
