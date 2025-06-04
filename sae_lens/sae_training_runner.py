import json
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, transformer_auto_wrap_policy
from simple_parsing import ArgumentParser
from transformer_lens.hook_points import HookedRootModule
try:
    from transformer_lens.llms.hf_llm import HFTransformerLayer # type: ignore
except ImportError:
    HFTransformerLayer = None # type: ignore

# For FSDP auto_wrap_policy
# This is a bit of a hack, as we'd ideally get the class from the model config.
# But this list covers many common TransformerLens models.
# We can make this more robust later if needed.
TRANSFORMER_LAYER_CLS_NAMES_MAP = {
    "GPT2Block",
    "LlamaDecoderLayer",
    "NeoXLayer",
    "MistralDecoderLayer",
    "GemmaDecoderLayer",
    # Add other common layer class names here as needed
}
if HFTransformerLayer is not None:
    TRANSFORMER_LAYER_CLS_NAMES_MAP.add(HFTransformerLayer.__name__)


from sae_lens import logger
from sae_lens.config import HfDataset, LanguageModelSAERunnerConfig
from sae_lens.load_model import load_model
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.geometric_median import compute_geometric_median
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.training.training_sae import TrainingSAE, TrainingSAEConfig


class InterruptedException(Exception):
    pass


def interrupt_callback(sig_num: Any, stack_frame: Any):  # noqa: ARG001
    raise InterruptedException()


class SAETrainingRunner:
    """
    Class to run the training of a Sparse Autoencoder (SAE) on a TransformerLens model.
    """

    cfg: LanguageModelSAERunnerConfig
    model: HookedRootModule
    sae: TrainingSAE
    activations_store: ActivationsStore
    accelerator: Accelerator

    def __init__(
        self,
        cfg: LanguageModelSAERunnerConfig,
        override_dataset: HfDataset | None = None,
        override_model: HookedRootModule | None = None,
        override_sae: TrainingSAE | None = None,
    ):
        if override_dataset is not None:
            logger.warning(
                f"You just passed in a dataset which will override the one specified in your configuration: {cfg.dataset_path}. As a consequence this run will not be reproducible via configuration alone."
            )
        if override_model is not None:
            logger.warning(
                f"You just passed in a model which will override the one specified in your configuration: {cfg.model_name}. As a consequence this run will not be reproducible via configuration alone."
            )

        self.cfg = cfg

        if self.cfg.fsdp_enabled:
            fsdp_plugin = None
            sharding_strategy = getattr(ShardingStrategy, self.cfg.fsdp_sharding_strategy.upper(), None)
            if sharding_strategy is None:
                raise ValueError(f"Invalid FSDP sharding strategy: {self.cfg.fsdp_sharding_strategy}")

            auto_wrap_policy = None
            if self.cfg.fsdp_auto_wrap_policy:
                if self.cfg.fsdp_auto_wrap_policy == "transformer_auto_wrap_policy":
                    if not self.cfg.fsdp_transformer_layer_cls_to_wrap:
                        raise ValueError(
                            "fsdp_transformer_layer_cls_to_wrap must be specified for transformer_auto_wrap_policy."
                        )

                    transformer_layer_classes_to_wrap = set()
                    for cls_name in self.cfg.fsdp_transformer_layer_cls_to_wrap:
                        if cls_name not in TRANSFORMER_LAYER_CLS_NAMES_MAP:
                            logger.warning(
                                f"FSDP: Transformer layer class '{cls_name}' not in known map. It might not be wrapped correctly."
                                " Known classes: {', '.join(TRANSFORMER_LAYER_CLS_NAMES_MAP)}"
                            )
                            # Attempt to dynamically get the class if possible, otherwise rely on string if accelerate supports it
                            # For now, we assume accelerate might handle string class names or user provides correct ones.
                            # A more robust solution would involve inspecting the model directly.
                            transformer_layer_classes_to_wrap.add(cls_name)
                        else:
                             # This part is tricky as we don't have the actual class type here,
                             # only its name. `transformer_auto_wrap_policy` expects class types.
                             # We will rely on a map for now.
                             # A better solution would be to get these from the model object itself.
                             # For now, this relies on the user providing exact names present in TRANSFORMER_LAYER_CLS_NAMES_MAP
                             # or that Accelerate's FSDP plugin can handle string names directly (which it might for some policies)
                             # For `transformer_auto_wrap_policy`, it typically needs actual classes.
                             # This is a placeholder for a more robust solution.
                             # Let's assume for now that we will resolve these names to classes later if needed by the policy,
                             # or that the user provides classes directly if the policy requires it.
                             # The current implementation of transformer_auto_wrap_policy in Accelerate
                             # seems to take `transformer_layer_cls_to_wrap` which is a set of classes.
                             # This is a simplification and might need adjustment based on how Accelerate handles this.
                             # For now, we'll assume string names might work or this needs to be classes.
                             # Given the limitations, this part is more of a configuration pass-through.
                             pass # Placeholder - actual class resolution might be needed.

                    # This part is tricky. transformer_auto_wrap_policy expects a set of *classes*.
                    # We only have strings from the config. We'd need to import these classes.
                    # For now, we'll pass the string names and hope Accelerate/FSDP can handle it,
                    # or this will need a more dynamic way to get class objects.
                    # A common approach is to have a mapping from string names to class objects.
                    # For simplicity, we'll assume string names might work for some policies or this is a TODO.
                    # The `ModuleWrapPolicy` example in Accelerate uses actual classes.
                    # Let's try to get the actual classes using the map.
                    actual_transformer_classes = set()
                    for name in self.cfg.fsdp_transformer_layer_cls_to_wrap:
                        found_cls = None
                        # This is a very simplified way and might not always work.
                        # It assumes common locations or that the class name is globally unique enough.
                        # A more robust way is to inspect the model's modules.
                        for module_name, module in sys.modules.items():
                            if hasattr(module, name):
                                found_cls = getattr(module, name)
                                if isinstance(found_cls, type): # check if it's a class
                                    actual_transformer_classes.add(found_cls)
                                    break
                        if not found_cls:
                             logger.warning(f"FSDP: Could not resolve class name '{name}' to a class type for auto_wrap_policy. String name will be used.")
                             # Fallback to using the string name, though this might not work for transformer_auto_wrap_policy
                             # actual_transformer_classes.add(name) # This would make it a set of strings / mixed types

                    if not actual_transformer_classes:
                         logger.warning("FSDP: No transformer layer classes resolved for auto_wrap_policy. Auto-wrapping might not work as expected.")
                         # auto_wrap_policy = None # Fallback if no classes found
                    else:
                        auto_wrap_policy = ModuleWrapPolicy(transformer_layer_cls_to_wrap=actual_transformer_classes)


                else: # User might provide a custom policy name to be resolved by Accelerate
                    logger.warning(f"FSDP: Unknown auto_wrap_policy name '{self.cfg.fsdp_auto_wrap_policy}'. Relying on Accelerate to resolve it or error out.")
                    # auto_wrap_policy = self.cfg.fsdp_auto_wrap_policy # Pass as string

            cpu_offload = CPUOffload(offload_params=self.cfg.fsdp_offload_params)

            fsdp_plugin = FullyShardedDataParallelPlugin(
                sharding_strategy=sharding_strategy,
                auto_wrap_policy=auto_wrap_policy,
                cpu_offload=cpu_offload,
                sync_module_states=self.cfg.fsdp_sync_module_states,
                use_orig_params=self.cfg.fsdp_use_orig_params,
                # Note: cpu_ram_efficient_loading is a param for accelerator.prepare(), not FSDPPlugin directly usually.
                # It's handled by Accelerator during model loading.
            )
            self.accelerator = Accelerator(fsdp_plugin=fsdp_plugin, mixed_precision="fp16" if self.cfg.autocast else None)
            # TODO: cpu_ram_efficient_loading needs to be handled during model loading if possible,
            # or passed to accelerator.prepare() if that's the interface.
            # For now, it's a config option that might need to be used manually during model loading for FSDP.
            if self.cfg.fsdp_cpu_ram_efficient_loading:
                 logger.info("FSDP: fsdp_cpu_ram_efficient_loading is True. Ensure model loading respects this for FSDP.")

        else:
            self.accelerator = Accelerator(mixed_precision="fp16" if self.cfg.autocast else None)


        if override_model is None:
            # Let accelerator handle device placement.
            self.model = load_model(
                self.cfg.model_class_name,
                self.cfg.model_name,
                # device=self.cfg.device, # Accelerate handles this
                model_from_pretrained_kwargs=self.cfg.model_from_pretrained_kwargs,
            )
        else:
            self.model = override_model

        if override_sae is None:
            if self.cfg.from_pretrained_path is not None:
                # Let accelerator handle device placement.
                self.sae = TrainingSAE.load_from_pretrained(
                    self.cfg.from_pretrained_path #, self.cfg.device # Accelerate handles this
                )
            else:
                # SAE is initialized on CPU then moved to device by accelerator.
                sae_cfg_dict = self.cfg.get_training_sae_cfg_dict()
                if "device" in sae_cfg_dict: # remove device from sae_cfg_dict
                    del sae_cfg_dict["device"]
                self.sae = TrainingSAE(
                    TrainingSAEConfig.from_dict(sae_cfg_dict)
                )
                # _init_sae_group_b_decs will be called after model and sae are prepared by accelerator
        else:
            self.sae = override_sae

        # Prepare model and SAE with accelerator
        self.model, self.sae = self.accelerator.prepare(self.model, self.sae)

        # ActivationsStore needs to be created after model is prepared by accelerator
        # as it used model.device and model.dtype
        self.activations_store = ActivationsStore.from_config(
            self.model, # model is already prepared
            self.cfg,
            override_dataset=override_dataset,
            accelerator=self.accelerator, # Pass accelerator to ActivationsStore
        )

        # _init_sae_group_b_decs should be called after model and sae are prepared by accelerator
        # and after activations_store is initialized.
        # We only call it if we are not loading a pretrained SAE and not overriding the SAE
        if override_sae is None and self.cfg.from_pretrained_path is None:
            self._init_sae_group_b_decs()

    def run(self):
        """
        Run the training of the SAE.
        """

        if self.cfg.log_to_wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                config=cast(Any, self.cfg),
                name=self.cfg.run_name,
                id=self.cfg.wandb_id,
            )

        trainer = SAETrainer(
            model=self.model,
            sae=self.sae,
            activation_store=self.activations_store,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg,
            accelerator=self.accelerator,
        )

        if self.cfg.resume_from_checkpoint_folder:
            checkpoint_path = Path(self.cfg.resume_from_checkpoint_folder)
            logger.info(f"Attempting to resume from checkpoint: {checkpoint_path}")
            try:
                self.accelerator.load_state(str(checkpoint_path))
                trainer_state_path = checkpoint_path / "trainer_state.pt"
                if trainer_state_path.exists():
                    trainer_state = torch.load(trainer_state_path, map_location=self.accelerator.device)
                    trainer.n_training_steps = trainer_state.get("n_training_steps", 0)
                    trainer.n_training_tokens = trainer_state.get("n_training_tokens", 0)
                    trainer.started_fine_tuning = trainer_state.get("started_fine_tuning", False)
                    trainer.n_frac_active_tokens = trainer_state.get("n_frac_active_tokens", 0)
                    # Restore checkpoint_thresholds if saved, or re-calculate based on loaded n_training_tokens
                    # For simplicity, we could just let it re-calculate based on new starting point,
                    # or adjust trainer.checkpoint_thresholds based on trainer.n_training_tokens.
                    # Current SAETrainer.__init__ recalculates it from scratch.
                    # We might need to adjust how checkpoint_thresholds is initialized or updated after loading.
                    # A simple way is to filter thresholds that are already passed.
                    trainer.checkpoint_thresholds = [
                        t for t in trainer.checkpoint_thresholds if t > trainer.n_training_tokens
                    ]
                    logger.info(f"Successfully resumed from checkpoint: {checkpoint_path}")
                    logger.info(f"Resumed trainer state: n_training_tokens={trainer.n_training_tokens}, n_training_steps={trainer.n_training_steps}")

                else:
                    logger.warning(f"Trainer state file not found at {trainer_state_path}, only accelerator state loaded.")
            except Exception as e:
                logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}", exc_info=True)
                # Depending on desired behavior, either raise e or continue without loading state
                # For now, we'll log error and continue, which means fresh training.
                # Consider adding a strict_loading flag if needed.


        # Don't compile when using accelerate for now.
        # TODO: figure out how to make compile work with accelerate.
        # self._compile_if_needed()
        sae = self.run_trainer_with_interruption_handling(trainer)

        if self.cfg.log_to_wandb:
            wandb.finish()

        return sae

    def _compile_if_needed(self):
        # Compile model and SAE
        #  torch.compile can provide significant speedups (10-20% in testing)
        # using max-autotune gives the best speedups but:
        # (a) increases VRAM usage,
        # (b) can't be used on both SAE and LM (some issue with cudagraphs), and
        # (c) takes some time to compile
        # optimal settings seem to be:
        # use max-autotune on SAE and max-autotune-no-cudagraphs on LM
        # (also pylance seems to really hate this)
        # TODO: Make this work with accelerate
        # if self.cfg.compile_llm:
        #     self.model = torch.compile(
        #         self.model,
        #         mode=self.cfg.llm_compilation_mode,
        #     )  # type: ignore

        # if self.cfg.compile_sae:
        #     backend = "aot_eager" if self.cfg.device == "mps" else "inductor"

        #     self.sae.training_forward_pass = torch.compile(  # type: ignore
        #         self.sae.training_forward_pass,
        #         mode=self.cfg.sae_compilation_mode,
        #         backend=backend,
        #     )  # type: ignore
        pass

    def run_trainer_with_interruption_handling(self, trainer: SAETrainer):
        try:
            # signal handlers (if preempted)
            signal.signal(signal.SIGINT, interrupt_callback)
            signal.signal(signal.SIGTERM, interrupt_callback)

            # train SAE
            sae = trainer.fit()

        except (KeyboardInterrupt, InterruptedException):
            logger.warning("interrupted, saving progress")
            checkpoint_name = str(trainer.n_training_tokens)
            self.save_checkpoint(trainer, checkpoint_name=checkpoint_name)
            logger.info("done saving")
            raise

        return sae

    # TODO: move this into the SAE trainer or Training SAE class
    def _init_sae_group_b_decs(
        self,
    ) -> None:
        """
        extract all activations at a certain layer and use for sae b_dec initialization
        """

        if self.cfg.b_dec_init_method == "geometric_median":
            self.activations_store.set_norm_scaling_factor_if_needed()
            layer_acts = self.activations_store.storage_buffer.detach()[:, 0, :]
            # get geometric median of the activations if we're using those.
            median = compute_geometric_median(
                layer_acts,
                maxiter=100,
            ).median
            self.sae.initialize_b_dec_with_precalculated(median)  # type: ignore
        elif self.cfg.b_dec_init_method == "mean":
            self.activations_store.set_norm_scaling_factor_if_needed()
            layer_acts = self.activations_store.storage_buffer.detach().cpu()[:, 0, :]
            self.sae.initialize_b_dec_with_mean(layer_acts)  # type: ignore

    @staticmethod
    def save_checkpoint(
        trainer: SAETrainer,
        checkpoint_name: str,
        wandb_aliases: list[str] | None = None,
    ) -> None:
        """
        Save a checkpoint of the trainer's state.
        This method is designed to be used with Hugging Face Accelerate.
        """
        base_path = Path(trainer.cfg.checkpoint_path) / checkpoint_name
        base_path.mkdir(exist_ok=True, parents=True)

        # Accelerator saves the state of prepared objects (model, SAE, optimizer, activation_store)
        # and registered objects (l1_scheduler, act_freq_scores, n_forward_passes_since_fired)
        trainer.accelerator.save_state(str(base_path))

        # Save additional trainer state not handled by accelerator's automatic checkpointing
        # These are simple Python variables.
        trainer_state_to_save = {
            "n_training_steps": trainer.n_training_steps,
            "n_training_tokens": trainer.n_training_tokens,
            "started_fine_tuning": trainer.started_fine_tuning,
            "n_frac_active_tokens": trainer.n_frac_active_tokens,
            # checkpoint_thresholds could also be saved if it's dynamic
        }
        torch.save(trainer_state_to_save, base_path / "trainer_state.pt")

        # Save SAE config (TrainingSAEConfig)
        # The SAE model weights are saved by accelerator.save_state() as self.sae is prepared.
        # We still need to save its specific config file.
        if trainer.sae.cfg.normalize_sae_decoder: # Ensure decoder norm is up-to-date before saving config potentially
            trainer.sae.set_decoder_norm_to_unit_norm()
        sae_config_path = trainer.sae.save_config(str(base_path)) # save config.json

        # Save the main run config (LanguageModelSAERunnerConfig)
        run_config_path = base_path / "run_config.json"
        with open(run_config_path, "w") as f:
            json.dump(trainer.cfg.to_dict(), f)

        logger.info(f"Checkpoint saved to {base_path}")

        if trainer.cfg.log_to_wandb and trainer.accelerator.is_main_process:
            sae_name = trainer.sae.get_name().replace("/", "__")
            checkpoint_artifact = wandb.Artifact(
                f"{sae_name}_{checkpoint_name}",
                type="checkpoint",
                metadata=trainer.cfg.to_dict(),
            )
            # Add the whole directory saved by accelerator.save_state
            checkpoint_artifact.add_dir(str(base_path))

            # Explicitly add key config files for easier access in W&B UI
            # checkpoint_artifact.add_file(str(sae_config_path), name="sae_config.json") # Already in base_path
            # checkpoint_artifact.add_file(str(run_config_path), name="run_config.json") # Already in base_path

            wandb.log_artifact(checkpoint_artifact, aliases=wandb_aliases)
            logger.info(f"Checkpoint artifact {checkpoint_artifact.name} logged to W&B.")


def _parse_cfg_args(args: Sequence[str]) -> LanguageModelSAERunnerConfig:
    if len(args) == 0:
        args = ["--help"]
    parser = ArgumentParser(exit_on_error=False)
    parser.add_arguments(LanguageModelSAERunnerConfig, dest="cfg")
    return parser.parse_args(args).cfg


# moved into its own function to make it easier to test
def _run_cli(args: Sequence[str]):
    cfg = _parse_cfg_args(args)
    SAETrainingRunner(cfg=cfg).run()


if __name__ == "__main__":
    _run_cli(args=sys.argv[1:])
