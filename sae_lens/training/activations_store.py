from __future__ import annotations

import contextlib
import json
import os
import warnings
from collections.abc import Generator, Iterator, Sequence
from typing import Any, Literal, cast

import datasets
import numpy as np
import torch
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from jaxtyping import Float, Int
from requests import HTTPError
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from accelerate import Accelerator # Added for FSDP

from sae_lens import logger
from sae_lens.config import (
    DTYPE_MAP,
    CacheActivationsRunnerConfig,
    HfDataset,
    LanguageModelSAERunnerConfig,
)
from sae_lens.sae import SAE
from sae_lens.tokenization_and_batching import concat_and_batch_sequences


# TODO: Make an activation store config class to be consistent with the rest of the code.
class ActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs.
    """

    model: HookedRootModule
    dataset: HfDataset
    cached_activations_path: str | None
    cached_activation_dataset: Dataset | None = None
    tokens_column: Literal["tokens", "input_ids", "text", "problem"]
    hook_name: str
    hook_layer: int
    hook_head_index: int | None
    _dataloader: Iterator[Any] | None = None
    _storage_buffer: torch.Tensor | None = None # This will store (activations, tokens_if_present)
    exclude_special_tokens: torch.Tensor | None = None
    device: torch.device # Device for the storage buffer (usually CPU)
    accelerator: Accelerator | None = None # For distributed activation generation

    @classmethod
    def from_cache_activations(
        cls,
        model: HookedRootModule,
        cfg: CacheActivationsRunnerConfig,
    ) -> ActivationsStore:
        """
        Public api to create an ActivationsStore from a cached activations dataset.
        """
        return cls(
            cached_activations_path=cfg.new_cached_activations_path,
            dtype=cfg.dtype,
            hook_name=cfg.hook_name,
            hook_layer=cfg.hook_layer,
            context_size=cfg.context_size,
            d_in=cfg.d_in,
            n_batches_in_buffer=cfg.n_batches_in_buffer,
            total_training_tokens=cfg.training_tokens,
            store_batch_size_prompts=cfg.model_batch_size,  # get_buffer
            train_batch_size_tokens=cfg.model_batch_size,  # dataloader
            seqpos_slice=(None,),
            device=torch.device(cfg.device),  # since we're sending these to SAE
            # NOOP
            prepend_bos=False,
            hook_head_index=None,
            dataset=cfg.dataset_path,
            streaming=False,
            model=model,
            normalize_activations="none",
            model_kwargs=None,
            autocast_lm=False,
            dataset_trust_remote_code=None,
            exclude_special_tokens=None,
            accelerator=None, # Added
        )

    @classmethod
    def from_config(
        cls,
        model: HookedRootModule,
        cfg: LanguageModelSAERunnerConfig | CacheActivationsRunnerConfig,
        override_dataset: HfDataset | None = None,
        accelerator: Accelerator | None = None, # Added
    ) -> ActivationsStore:
        if isinstance(cfg, CacheActivationsRunnerConfig):
            # Note: CacheActivationsRunnerConfig does not have fsdp settings, so accelerator is not passed here.
            # If cached activations were generated in an FSDP context, they should already be gathered.
            return cls.from_cache_activations(model, cfg)

        cached_activations_path = cfg.cached_activations_path
        # set cached_activations_path to None if we're not using cached activations
        if (
            isinstance(cfg, LanguageModelSAERunnerConfig)
            and not cfg.use_cached_activations
        ):
            cached_activations_path = None

        if override_dataset is None and cfg.dataset_path == "":
            raise ValueError(
                "You must either pass in a dataset or specify a dataset_path in your configutation."
            )

        device = torch.device(cfg.act_store_device)
        exclude_special_tokens = cfg.exclude_special_tokens
        if exclude_special_tokens is False:
            exclude_special_tokens = None
        if exclude_special_tokens is True:
            exclude_special_tokens = _get_special_token_ids(model.tokenizer)  # type: ignore
        if exclude_special_tokens is not None:
            exclude_special_tokens = torch.tensor(
                exclude_special_tokens, dtype=torch.long, device=device
            )
        return cls(
            model=model,
            dataset=override_dataset or cfg.dataset_path,
            streaming=cfg.streaming,
            hook_name=cfg.hook_name,
            hook_layer=cfg.hook_layer,
            hook_head_index=cfg.hook_head_index,
            context_size=cfg.context_size,
            d_in=cfg.d_in,
            n_batches_in_buffer=cfg.n_batches_in_buffer,
            total_training_tokens=cfg.training_tokens,
            store_batch_size_prompts=cfg.store_batch_size_prompts,
            train_batch_size_tokens=cfg.train_batch_size_tokens,
            prepend_bos=cfg.prepend_bos,
            normalize_activations=cfg.normalize_activations,
            device=device,
            dtype=cfg.dtype,
            cached_activations_path=cached_activations_path,
            model_kwargs=cfg.model_kwargs,
            autocast_lm=cfg.autocast_lm,
            dataset_trust_remote_code=cfg.dataset_trust_remote_code,
            seqpos_slice=cfg.seqpos_slice,
            exclude_special_tokens=exclude_special_tokens,
            accelerator=accelerator, # Added
        )

    @classmethod
    def from_sae(
        cls,
        model: HookedRootModule,
        sae: SAE,
        context_size: int | None = None,
        dataset: HfDataset | str | None = None,
        streaming: bool = True,
        store_batch_size_prompts: int = 8,
        n_batches_in_buffer: int = 8,
        train_batch_size_tokens: int = 4096,
        total_tokens: int = 10**9,
        device: str = "cpu",
    ) -> ActivationsStore:
        return cls(
            model=model,
            dataset=sae.cfg.dataset_path if dataset is None else dataset,
            d_in=sae.cfg.d_in,
            hook_name=sae.cfg.hook_name,
            hook_layer=sae.cfg.hook_layer,
            hook_head_index=sae.cfg.hook_head_index,
            context_size=sae.cfg.context_size if context_size is None else context_size,
            prepend_bos=sae.cfg.prepend_bos,
            streaming=streaming,
            store_batch_size_prompts=store_batch_size_prompts,
            train_batch_size_tokens=train_batch_size_tokens,
            n_batches_in_buffer=n_batches_in_buffer,
            total_training_tokens=total_tokens,
            normalize_activations=sae.cfg.normalize_activations,
            dataset_trust_remote_code=sae.cfg.dataset_trust_remote_code,
            dtype=sae.cfg.dtype,
            device=torch.device(device), # storage device
            seqpos_slice=sae.cfg.seqpos_slice,
            accelerator=None, # Not typically used when loading from SAE directly like this
        )

    def __init__(
        self,
        model: HookedRootModule,
        dataset: HfDataset | str,
        streaming: bool,
        hook_name: str,
        hook_layer: int,
        hook_head_index: int | None,
        context_size: int,
        d_in: int,
        n_batches_in_buffer: int,
        total_training_tokens: int,
        store_batch_size_prompts: int,
        train_batch_size_tokens: int,
        prepend_bos: bool,
        normalize_activations: str,
        device: torch.device,
        dtype: str,
        cached_activations_path: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        autocast_lm: bool = False,
        dataset_trust_remote_code: bool | None = None,
        seqpos_slice: tuple[int | None, ...] = (None,),
        exclude_special_tokens: torch.Tensor | None = None,
        accelerator: Accelerator | None = None, # Added
    ):
        self.model = model # This model might be FSDP wrapped by the runner
        self.accelerator = accelerator
        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs = model_kwargs
        self.dataset = (
            load_dataset(
                dataset,
                split="train",
                streaming=streaming,
                trust_remote_code=dataset_trust_remote_code,  # type: ignore
            )
            if isinstance(dataset, str)
            else dataset
        )

        if isinstance(dataset, (Dataset, DatasetDict)):
            self.dataset = cast(Dataset | DatasetDict, self.dataset)
            n_samples = len(self.dataset)

            if n_samples < total_training_tokens:
                warnings.warn(
                    f"The training dataset contains fewer samples ({n_samples}) than the number of samples required by your training configuration ({total_training_tokens}). This will result in multiple training epochs and some samples being used more than once."
                )

        self.hook_name = hook_name
        self.hook_layer = hook_layer
        self.hook_head_index = hook_head_index
        self.context_size = context_size
        self.d_in = d_in
        self.n_batches_in_buffer = n_batches_in_buffer
        self.half_buffer_size = n_batches_in_buffer // 2
        self.total_training_tokens = total_training_tokens
        self.store_batch_size_prompts = store_batch_size_prompts
        self.train_batch_size_tokens = train_batch_size_tokens
        self.prepend_bos = prepend_bos
        self.normalize_activations = normalize_activations
        # self.device is for the activation *store* (buffer), typically CPU.
        # The model itself will run on self.accelerator.device if accelerator is used.
        self.device = torch.device(device)
        self.dtype = DTYPE_MAP[dtype]
        self.cached_activations_path = cached_activations_path
        self.autocast_lm = autocast_lm
        self.seqpos_slice = seqpos_slice
        self.exclude_special_tokens = exclude_special_tokens

        self.n_dataset_processed = 0

        self.estimated_norm_scaling_factor = None

        # Check if dataset is tokenized
        dataset_sample = next(iter(self.dataset))

        # check if it's tokenized
        if "tokens" in dataset_sample:
            self.is_dataset_tokenized = True
            self.tokens_column = "tokens"
        elif "input_ids" in dataset_sample:
            self.is_dataset_tokenized = True
            self.tokens_column = "input_ids"
        elif "text" in dataset_sample:
            self.is_dataset_tokenized = False
            self.tokens_column = "text"
        elif "problem" in dataset_sample:
            self.is_dataset_tokenized = False
            self.tokens_column = "problem"
        else:
            raise ValueError(
                "Dataset must have a 'tokens', 'input_ids', 'text', or 'problem' column."
            )
        if self.is_dataset_tokenized:
            ds_context_size = len(dataset_sample[self.tokens_column])
            if ds_context_size < self.context_size:
                raise ValueError(
                    f"""pretokenized dataset has context_size {ds_context_size}, but the provided context_size is {self.context_size}.
                    The context_size {ds_context_size} is expected to be larger than or equal to the provided context size {self.context_size}."""
                )
            if self.context_size != ds_context_size:
                warnings.warn(
                    f"""pretokenized dataset has context_size {ds_context_size}, but the provided context_size is {self.context_size}. Some data will be discarded in this case.""",
                    RuntimeWarning,
                )
            # TODO: investigate if this can work for iterable datasets, or if this is even worthwhile as a perf improvement
            if hasattr(self.dataset, "set_format"):
                self.dataset.set_format(type="torch", columns=[self.tokens_column])  # type: ignore

            if (
                isinstance(dataset, str)
                and hasattr(model, "tokenizer")
                and model.tokenizer is not None
            ):
                validate_pretokenized_dataset_tokenizer(
                    dataset_path=dataset,
                    model_tokenizer=model.tokenizer,  # type: ignore
                )
        else:
            warnings.warn(
                "Dataset is not tokenized. Pre-tokenizing will improve performance and allows for more control over special tokens. See https://jbloomaus.github.io/SAELens/training_saes/#pretokenizing-datasets for more info."
            )

        self.iterable_sequences = self._iterate_tokenized_sequences()

        self.cached_activation_dataset = self.load_cached_activation_dataset()

        # TODO add support for "mixed loading" (ie use cache until you run out, then switch over to streaming from HF)

    def _iterate_raw_dataset(
        self,
    ) -> Generator[torch.Tensor | list[int] | str, None, None]:
        """
        Helper to iterate over the dataset while incrementing n_dataset_processed
        """
        for row in self.dataset:
            # typing datasets is difficult
            yield row[self.tokens_column]  # type: ignore
            self.n_dataset_processed += 1

    def _iterate_raw_dataset_tokens(self) -> Generator[torch.Tensor, None, None]:
        """
        Helper to create an iterator which tokenizes raw text from the dataset on the fly
        """
        for row in self._iterate_raw_dataset():
            tokens = (
                self.model.to_tokens(
                    row,
                    truncate=False,
                    move_to_device=False,  # we move to device below
                    prepend_bos=False,
                )  # type: ignore
                .squeeze(0)
                .to(self.device)
            )
            if len(tokens.shape) != 1:
                raise ValueError(f"tokens.shape should be 1D but was {tokens.shape}")
            yield tokens

    def _iterate_tokenized_sequences(self) -> Generator[torch.Tensor, None, None]:
        """
        Generator which iterates over full sequence of context_size tokens
        """
        # If the datset is pretokenized, we will slice the dataset to the length of the context window if needed. Otherwise, no further processing is needed.
        # We assume that all necessary BOS/EOS/SEP tokens have been added during pretokenization.
        if self.is_dataset_tokenized:
            for row in self._iterate_raw_dataset():
                yield torch.tensor(
                    row[
                        : self.context_size
                    ],  # If self.context_size = None, this line simply returns the whole row
                    dtype=torch.long,
                    device=self.device,
                    requires_grad=False,
                )
        # If the dataset isn't tokenized, we'll tokenize, concat, and batch on the fly
        else:
            tokenizer = getattr(self.model, "tokenizer", None)
            bos_token_id = None if tokenizer is None else tokenizer.bos_token_id
            yield from concat_and_batch_sequences(
                tokens_iterator=self._iterate_raw_dataset_tokens(),
                context_size=self.context_size,
                begin_batch_token_id=(bos_token_id if self.prepend_bos else None),
                begin_sequence_token_id=None,
                sequence_separator_token_id=(
                    bos_token_id if self.prepend_bos else None
                ),
            )

    def load_cached_activation_dataset(self) -> Dataset | None:
        """
        Load the cached activation dataset from disk.

        - If cached_activations_path is set, returns Huggingface Dataset else None
        - Checks that the loaded dataset has current has activations for hooks in config and that shapes match.
        """
        if self.cached_activations_path is None:
            return None

        assert self.cached_activations_path is not None  # keep pyright happy
        # Sanity check: does the cache directory exist?
        if not os.path.exists(self.cached_activations_path):
            raise FileNotFoundError(
                f"Cache directory {self.cached_activations_path} does not exist. "
                "Consider double-checking your dataset, model, and hook names."
            )

        # ---
        # Actual code
        activations_dataset = datasets.load_from_disk(self.cached_activations_path)
        columns = [self.hook_name]
        if "token_ids" in activations_dataset.column_names:
            columns.append("token_ids")
        activations_dataset.set_format(
            type="torch", columns=columns, device=self.device, dtype=self.dtype
        )
        self.current_row_idx = 0  # idx to load next batch from
        # ---

        assert isinstance(activations_dataset, Dataset)

        # multiple in hooks future
        if not set([self.hook_name]).issubset(activations_dataset.column_names):
            raise ValueError(
                f"loaded dataset does not include hook activations, got {activations_dataset.column_names}"
            )

        if activations_dataset.features[self.hook_name].shape != (
            self.context_size,
            self.d_in,
        ):
            raise ValueError(
                f"Given dataset of shape {activations_dataset.features[self.hook_name].shape} does not match context_size ({self.context_size}) and d_in ({self.d_in})"
            )

        return activations_dataset

    def set_norm_scaling_factor_if_needed(self):
        if (
            self.normalize_activations == "expected_average_only_in"
            and self.estimated_norm_scaling_factor is None
        ):
            self.estimated_norm_scaling_factor = self.estimate_norm_scaling_factor()

    def apply_norm_scaling_factor(self, activations: torch.Tensor) -> torch.Tensor:
        if self.estimated_norm_scaling_factor is None:
            raise ValueError(
                "estimated_norm_scaling_factor is not set, call set_norm_scaling_factor_if_needed() first"
            )
        return activations * self.estimated_norm_scaling_factor

    def unscale(self, activations: torch.Tensor) -> torch.Tensor:
        if self.estimated_norm_scaling_factor is None:
            raise ValueError(
                "estimated_norm_scaling_factor is not set, call set_norm_scaling_factor_if_needed() first"
            )
        return activations / self.estimated_norm_scaling_factor

    def get_norm_scaling_factor(self, activations: torch.Tensor) -> torch.Tensor:
        return (self.d_in**0.5) / activations.norm(dim=-1).mean()

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, n_batches_for_norm_estimate: int = int(1e3)):
        norms_per_batch = []
        for _ in tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            # temporalily set estimated_norm_scaling_factor to 1.0 so the dataloader works
            self.estimated_norm_scaling_factor = 1.0
            acts = self.next_batch()[:, 0]
            self.estimated_norm_scaling_factor = None
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        return np.sqrt(self.d_in) / mean_norm

    def shuffle_input_dataset(self, seed: int, buffer_size: int = 1):
        """
        This applies a shuffle to the huggingface dataset that is the input to the activations store. This
        also shuffles the shards of the dataset, which is especially useful for evaluating on different
        sections of very large streaming datasets. Buffer size is only relevant for streaming datasets.
        The default buffer_size of 1 means that only the shard will be shuffled; larger buffer sizes will
        additionally shuffle individual elements within the shard.
        """
        if isinstance(self.dataset, IterableDataset):
            self.dataset = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)
        else:
            self.dataset = self.dataset.shuffle(seed=seed)
        self.iterable_dataset = iter(self.dataset)

    def reset_input_dataset(self):
        """
        Resets the input dataset iterator to the beginning.
        """
        self.iterable_dataset = iter(self.dataset)

    @property
    def storage_buffer(self) -> torch.Tensor:
        if self._storage_buffer is None:
            self._storage_buffer = _filter_buffer_acts(
                self.get_buffer(self.half_buffer_size), self.exclude_special_tokens
            )

        return self._storage_buffer

    @property
    def dataloader(self) -> Iterator[Any]:
        if self._dataloader is None:
            self._dataloader = self.get_data_loader()
        return self._dataloader

    def get_batch_tokens(
        self, batch_size: int | None = None, raise_at_epoch_end: bool = False
    ):
        """
        Streams a batch of tokens from a dataset.

        If raise_at_epoch_end is true we will reset the dataset at the end of each epoch and raise a StopIteration. Otherwise we will reset silently.
        """
        if not batch_size:
            batch_size = self.store_batch_size_prompts
        sequences = []
        # the sequences iterator yields fully formed tokens of size context_size, so we just need to cat these into a batch
        for _ in range(batch_size):
            try:
                sequences.append(next(self.iterable_sequences))
            except StopIteration:
                self.iterable_sequences = self._iterate_tokenized_sequences()
                if raise_at_epoch_end:
                    raise StopIteration(
                        f"Ran out of tokens in dataset after {self.n_dataset_processed} samples, beginning the next epoch."
                    )
                sequences.append(next(self.iterable_sequences))

        # If accelerator is used, move to its device, otherwise _get_model_device
        target_device = self.accelerator.device if self.accelerator and hasattr(self.accelerator, "device") else _get_model_device(self.model)
        return torch.stack(sequences, dim=0).to(target_device)

    @torch.no_grad()
    def get_activations(self, batch_tokens: torch.Tensor): # batch_tokens are on accelerator.device or model's device
        """
        Returns activations of shape (batches, context, num_layers, d_in)

        d_in may result from a concatenated head dimension.
        """

        # Setup autocast if using
        if self.autocast_lm:
            autocast_if_enabled = torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.autocast_lm,
            )
        else:
            autocast_if_enabled = contextlib.nullcontext()

        with autocast_if_enabled:
            layerwise_activations_cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_name],
                stop_at_layer=self.hook_layer + 1,
                prepend_bos=False,
                **self.model_kwargs,
            )[1]

        layerwise_activations = layerwise_activations_cache[self.hook_name][
            :, slice(*self.seqpos_slice)
        ]

        n_batches, n_context = layerwise_activations.shape[:2]

        stacked_activations = torch.zeros((n_batches, n_context, 1, self.d_in), device=layerwise_activations.device) # Keep on same device as input for now

        if self.hook_head_index is not None:
            processed_layer_acts = layerwise_activations[
                :, :, self.hook_head_index
            ]
        elif layerwise_activations.ndim > 3:  # if we have a head dimension
            try:
                processed_layer_acts = layerwise_activations.view(
                    n_batches, n_context, -1
                )
            except RuntimeError as e:
                logger.error(f"Error during view operation: {e}")
                logger.info("Attempting to use reshape instead...")
                processed_layer_acts = layerwise_activations.reshape(
                    n_batches, n_context, -1
                )
        else:
            processed_layer_acts = layerwise_activations

        stacked_activations[:, :, 0] = processed_layer_acts

        # Gather activations if running in a distributed environment
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            # stacked_activations is currently (local_batch_on_this_rank, context, 1, d_in)
            # We need to gather across the batch dimension.
            # accelerator.gather reshapes based on the first dimension.
            # Let's reshape to (local_batch_on_this_rank, features) then gather, then reshape back.
            original_shape = stacked_activations.shape
            local_batch_size = original_shape[0]
            features_dim = original_shape[1] * original_shape[2] * original_shape[3]

            reshaped_for_gather = stacked_activations.reshape(local_batch_size, features_dim)
            gathered_reshaped = self.accelerator.gather(reshaped_for_gather)

            # Reshape back to (global_batch_size, context, 1, d_in)
            # The first dimension of gathered_reshaped is now global_batch_size
            global_batch_size = gathered_reshaped.shape[0]
            expected_shape = (global_batch_size, original_shape[1], original_shape[2], original_shape[3])
            stacked_activations = gathered_reshaped.reshape(expected_shape)

        return stacked_activations # On each process, this is now the *full* batch of activations

    def _load_buffer_from_cached(
        self,
        total_size: int,
        context_size: int,
        num_layers: int,
        d_in: int,
        raise_on_epoch_end: bool,
    ) -> tuple[
        Float[torch.Tensor, "(total_size context_size) num_layers d_in"],
        Int[torch.Tensor, "(total_size context_size)"] | None,
    ]:
        """
        Loads `total_size` activations from `cached_activation_dataset`.
        If running in a distributed environment, each process loads a shard of the data,
        and then the activations are gathered.
        The dataset has columns for each hook_name,
        each containing activations of shape (context_size, d_in).
        Raises StopIteration if the dataset is exhausted and raise_on_epoch_end is True.
        """
        assert self.cached_activation_dataset is not None
        hook_names = [self.hook_name] # In future, could be a list of multiple hook names
        if not set(hook_names).issubset(self.cached_activation_dataset.column_names):
            raise ValueError(
                f"Missing columns in dataset. Expected {hook_names}, "
                f"got {self.cached_activation_dataset.column_names}."
            )

        dataset_len = len(self.cached_activation_dataset)

        # Determine the global range of data to load for this buffer refill
        global_start_idx = self.current_row_idx

        # Calculate the actual number of samples we can load in this pass globally
        if global_start_idx >= dataset_len:
            self.current_row_idx = 0 # Reset for next epoch
            if raise_on_epoch_end:
                raise StopIteration("Dataset exhausted at start of buffer load.")
            global_start_idx = 0 # Start from beginning for non-epoch-raising case

        # Determine how many samples can be loaded globally for this iteration
        # This is total_size unless we are at the end of the dataset.
        num_globally_available_from_start_idx = dataset_len - global_start_idx
        current_global_load_size = min(total_size, num_globally_available_from_start_idx)

        if current_global_load_size <= 0: # Should not happen if global_start_idx was reset correctly
            self.current_row_idx = 0
            if raise_on_epoch_end:
                 raise StopIteration("No data to load, dataset possibly smaller than buffer or exhausted.")
            # If not raising, try to load from beginning again if possible
            global_start_idx = 0
            current_global_load_size = min(total_size, dataset_len)
            if current_global_load_size <= 0: # Still no data (e.g. empty dataset)
                return torch.empty((0, num_layers, d_in), dtype=self.dtype, device=self.device), None


        global_indices_to_load = list(range(global_start_idx, global_start_idx + current_global_load_size))

        acts_buffer_local_list = []
        token_ids_buffer_local = None # Placeholder for local token ids

        # Distributed loading logic vs single process
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            process_index = self.accelerator.process_index
            num_processes = self.accelerator.num_processes

            with self.accelerator.split_between_processes(global_indices_to_load, apply_padding=False) as local_indices_this_process_padded:
                # Slicing HF dataset with an empty list is problematic, ensure indices are present
                if not local_indices_this_process_padded:
                    local_ds_slice = None
                else:
                    # Datasets expect list of ints for slicing, not tensors.
                    local_indices_list = [int(i) for i in local_indices_this_process_padded]
                    local_ds_slice = self.cached_activation_dataset[local_indices_list]

            if local_ds_slice and len(local_ds_slice[self.hook_name]) > 0:
                for hook_name_iter in hook_names: # Usually just one hook_name
                    # _hook_buffer_local is (local_slice_len, context_size, d_in)
                    _hook_buffer_local = local_ds_slice[hook_name_iter]
                    acts_buffer_local_list.append(_hook_buffer_local)

                if "token_ids" in self.cached_activation_dataset.column_names:
                    token_ids_buffer_local = local_ds_slice["token_ids"] # (local_slice_len, context_size)

            # Prepare for gather - all tensors must be on the accelerator's device
            # And they must have the same shape on all processes for gather, or use gather_object
            # For tensor gather, if a process has no data, it should contribute an empty tensor of correct ndim & device.

            if acts_buffer_local_list: # If this process loaded some data
                local_acts_stacked = torch.stack(acts_buffer_local_list, dim=2).to(self.accelerator.device) # (local_slice_len, context_size, num_layers, d_in)
                original_shape_local = local_acts_stacked.shape
                local_slice_len = original_shape_local[0]
                features_dim = original_shape_local[1] * original_shape_local[2] * original_shape_local[3]
                reshaped_for_gather = local_acts_stacked.reshape(local_slice_len, features_dim)
            else: # This process had no data for this global batch part
                local_slice_len = 0 # For clarity
                features_dim = context_size * num_layers * d_in
                reshaped_for_gather = torch.empty((0, features_dim), dtype=self.dtype, device=self.accelerator.device)

            # Gather activations across all processes
            gathered_reshaped_acts = self.accelerator.gather(reshaped_for_gather)
            # Reshape back to (current_global_load_size, context_size, num_layers, d_in)
            acts_buffer = gathered_reshaped_acts.reshape(current_global_load_size, context_size, num_layers, d_in)

            if "token_ids" in self.cached_activation_dataset.column_names:
                if token_ids_buffer_local is not None and token_ids_buffer_local.nelement() > 0:
                    token_ids_buffer_local_on_device = token_ids_buffer_local.to(self.accelerator.device)
                    reshaped_tokens_for_gather = token_ids_buffer_local_on_device.reshape(local_slice_len, -1)
                else:
                    reshaped_tokens_for_gather = torch.empty((0, context_size), dtype=torch.long, device=self.accelerator.device)

                gathered_reshaped_tokens = self.accelerator.gather(reshaped_tokens_for_gather)
                token_ids_buffer = gathered_reshaped_tokens.reshape(current_global_load_size, context_size)
            else:
                token_ids_buffer = None

        else: # Single process logic
            ds_slice = self.cached_activation_dataset[global_indices_to_load]
            for hook_name_iter in hook_names:
                _hook_buffer = ds_slice[hook_name_iter]
                acts_buffer_local_list.append(_hook_buffer)

            acts_buffer = torch.stack(acts_buffer_local_list, dim=2) # (current_global_load_size, context_size, num_layers, d_in)

            if "token_ids" in self.cached_activation_dataset.column_names:
                token_ids_buffer = ds_slice["token_ids"] # (current_global_load_size, context_size)
            else:
                token_ids_buffer = None

        # Advance current_row_idx by the amount of data processed globally
        self.current_row_idx = global_start_idx + current_global_load_size
        if self.current_row_idx >= dataset_len: # Reset if reached or passed end
            self.current_row_idx = 0
            if raise_on_epoch_end and (global_start_idx + current_global_load_size) >= dataset_len :
                 raise StopIteration("Dataset exhausted during _load_buffer_from_cached.")

        # Reshape to (total_tokens_in_buffer_for_this_load, num_layers, d_in)
        # acts_buffer is already on self.device (CPU) if not distributed, or gathered to all devices then moved.
        # Ensure it's on the ActivationsStore's designated device (e.g. CPU)
        acts_buffer = acts_buffer.to(self.device)
        acts_buffer = acts_buffer.reshape(current_global_load_size * context_size, num_layers, d_in)

        if token_ids_buffer is not None:
            token_ids_buffer = token_ids_buffer.to(self.device)
            token_ids_buffer = token_ids_buffer.reshape(current_global_load_size * context_size)

        return acts_buffer, token_ids_buffer

    @torch.no_grad()
    def get_buffer(
        self,
        n_batches_in_buffer: int,
        raise_on_epoch_end: bool = False,
        shuffle: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Loads the next n_batches_in_buffer batches of activations into a tensor and returns it.

        The primary purpose here is maintaining a shuffling buffer.

        If raise_on_epoch_end is True, when the dataset it exhausted it will automatically refill the dataset and then raise a StopIteration so that the caller has a chance to react.
        """
        context_size = self.context_size
        training_context_size = len(range(context_size)[slice(*self.seqpos_slice)])
        batch_size = self.store_batch_size_prompts
        d_in = self.d_in
        total_size = batch_size * n_batches_in_buffer
        num_layers = 1

        if self.cached_activation_dataset is not None:
            return self._load_buffer_from_cached(
                total_size, context_size, num_layers, d_in, raise_on_epoch_end
            )

        refill_iterator = range(0, total_size, batch_size)
        # Initialize empty tensor buffer of the maximum required size with an additional dimension for layers
        new_buffer_activations = torch.zeros(
            (total_size, training_context_size, num_layers, d_in),
            dtype=self.dtype,  # type: ignore
            device=self.device,
        )
        new_buffer_token_ids = torch.zeros(
            (total_size, training_context_size),
            dtype=torch.long,
            device=self.device,
        )

        for refill_batch_idx_start in tqdm(
            refill_iterator, leave=False, desc="Refilling buffer"
        ):
            # get_batch_tokens already places on self.accelerator.device if accelerator is present
            refill_batch_tokens = self.get_batch_tokens(
                raise_at_epoch_end=raise_on_epoch_end
            )
            # get_activations will handle gathering if distributed.
            # refill_activations will be the full global batch on all processes.
            refill_activations = self.get_activations(refill_batch_tokens)

            # The buffer is on self.device (e.g. CPU). Move activations there.
            # If FSDP is used, refill_activations might be on each rank's GPU.
            # Ensure it's moved to the buffer's device.
            refill_activations = refill_activations.to(self.device)

            # Store the full batch of activations.
            # If store_batch_size_prompts was global batch size, this is correct.
            # If it was per-device, then this needs adjustment based on num_processes.
            # Assuming self.store_batch_size_prompts is the global batch size for LLM.
            current_global_batch_size = refill_activations.shape[0]


            # Only the main process should write to the buffer if we want to avoid redundant storage
            # and ensure correct batch accounting for the final SAE training dataloader,
            # UNLESS all processes will maintain an identical buffer and dataloader.
            # For now, let's assume all processes build an identical buffer.
            # This simplifies dataloading for the SAE later if it's also distributed (e.g. DDP).
            # The new_buffer_activations is sized for total_size = self.store_batch_size_prompts * n_batches_in_buffer.
            # If store_batch_size_prompts means global batch size, then this is fine.

            # The refill_iterator is based on self.store_batch_size_prompts.
            # If store_batch_size_prompts = global batch size for LLM fwd pass.
            # And refill_activations is (global_batch_size, context, num_layers, d_in)
            # Then this slice is correct.
            idx_end = refill_batch_idx_start + current_global_batch_size
            if idx_end > new_buffer_activations.shape[0]: # Handle cases where the last batch might be smaller
                idx_end = new_buffer_activations.shape[0]
                current_global_batch_size = idx_end - refill_batch_idx_start
                refill_activations = refill_activations[:current_global_batch_size]


            new_buffer_activations[
                refill_batch_idx_start : idx_end, ...
            ] = refill_activations # refill_activations is (global_batch_size, ...)

            # Corresponding tokens for this global batch
            # refill_batch_tokens was (global_batch_size, original_context_size)
            # We need to slice it like activations were sliced by seqpos_slice
            sliced_tokens = refill_batch_tokens[:, slice(*self.seqpos_slice)]
            # And then move to the buffer's device.
            new_buffer_token_ids[
                refill_batch_idx_start : idx_end, ...
            ] = sliced_tokens[:current_global_batch_size].to(self.device)


        new_buffer_activations = new_buffer_activations.reshape(-1, num_layers, d_in)
        new_buffer_token_ids = new_buffer_token_ids.reshape(-1)

        # Shuffle on all processes identically if they all have the full buffer.
        # Requires synchronized random state or ensuring torch.randperm is same if seed is managed.
        # For simplicity, if accelerator is present, let main process shuffle and then broadcast,
        # or let each process shuffle (if data is identical, shuffle will be too with same seed).
        # For now, assume shuffle happens on all processes over identical data.
        if shuffle:
            # Ensure consistent shuffling across processes if they all have the same data.
            # This usually means setting the same seed before this operation if not already done globally.
            # Or, main process shuffles and broadcasts indices.
            # For now, let's assume simple identical shuffle due to identical data.
            new_buffer_activations, new_buffer_token_ids = permute_together(
                [new_buffer_activations, new_buffer_token_ids]
            )

        # every buffer should be normalized:
        if self.normalize_activations == "expected_average_only_in":
            new_buffer_activations = self.apply_norm_scaling_factor(
                new_buffer_activations
            )

        return (
            new_buffer_activations,
            new_buffer_token_ids,
        )

    def get_data_loader(
        self,
    ) -> Iterator[Any]:
        """
        Return a torch.utils.dataloader which you can get batches from.

        Should automatically refill the buffer when it gets to n % full.
        (better mixing if you refill and shuffle regularly).

        """

        batch_size = self.train_batch_size_tokens

        try:
            new_samples = _filter_buffer_acts(
                self.get_buffer(self.half_buffer_size, raise_on_epoch_end=True),
                self.exclude_special_tokens,
            )
        except StopIteration:
            warnings.warn(
                "All samples in the training dataset have been exhausted, we are now beginning a new epoch with the same samples."
            )
            self._storage_buffer = (
                None  # dump the current buffer so samples do not leak between epochs
            )
            try:
                new_samples = _filter_buffer_acts(
                    self.get_buffer(self.half_buffer_size),
                    self.exclude_special_tokens,
                )
            except StopIteration:
                raise ValueError(
                    "We were unable to fill up the buffer directly after starting a new epoch. This could indicate that there are less samples in the dataset than are required to fill up the buffer. Consider reducing batch_size or n_batches_in_buffer. "
                )

        # 1. # create new buffer by mixing stored and new buffer
        mixing_buffer = torch.cat(
            [new_samples, self.storage_buffer],
            dim=0,
        )

        mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]

        # 2.  put 50 % in storage
        self._storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

        # 3. put other 50 % in a dataloader
        return iter(
            DataLoader(
                # TODO: seems like a typing bug?
                cast(Any, mixing_buffer[mixing_buffer.shape[0] // 2 :]),
                batch_size=batch_size,
                shuffle=True,
            )
        )

    def next_batch(self) -> torch.Tensor:
        """
        Get the next batch from the current DataLoader.
        If the DataLoader is exhausted, refill the buffer and create a new DataLoader.
        """
        try:
            # Try to get the next batch
            return next(self.dataloader)
        except StopIteration:
            # If the DataLoader is exhausted, create a new one
            self._dataloader = self.get_data_loader()
            return next(self.dataloader)

    def state_dict(self) -> dict[str, Any]:
        # _storage_buffer directly stores the filtered activation tensor,
        # not a tuple (activations, tokens). Tokens are used for filtering but not stored in this attribute.
        result = {
            "n_dataset_processed": self.n_dataset_processed,
            "current_row_idx": self.current_row_idx,
            "estimated_norm_scaling_factor": self.estimated_norm_scaling_factor,
            "_storage_buffer": self._storage_buffer, # This is Tensor | None
        }
        return result

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.n_dataset_processed = state_dict.get("n_dataset_processed", 0)
        self.current_row_idx = state_dict.get("current_row_idx", 0)
        self.estimated_norm_scaling_factor = state_dict.get("estimated_norm_scaling_factor")

        # Load the activations tensor into _storage_buffer
        self._storage_buffer = state_dict.get("_storage_buffer") # This will be Tensor | None

        # Crucially, reset the dataloader so it's recreated with the new buffer if needed
        self._dataloader = None

        # Resetting iterable_sequences to force re-evaluation from the new n_dataset_processed/current_row_idx.
        # This is a simplification. True state restoration of arbitrary generators is hard.
        # For map-style datasets, _iterate_raw_dataset will use n_dataset_processed.
        # For IterableDatasets, it will restart the stream; n_dataset_processed is for accounting.
        # If cached_activation_dataset is used, current_row_idx will take effect when _load_buffer_from_cached is called.
        if self.cached_activations_path is None: # Only reset iterable_sequences if not using cache primarily
             self.iterable_sequences = self._iterate_tokenized_sequences()
        # If using cached_activations, current_row_idx is the primary cursor. iterable_sequences might not be used.

    def save(self, file_path: str):
        """save the state dict to a file in safetensors format"""
        save_file(self.state_dict(), file_path)


def validate_pretokenized_dataset_tokenizer(
    dataset_path: str, model_tokenizer: PreTrainedTokenizerBase
) -> None:
    """
    Helper to validate that the tokenizer used to pretokenize the dataset matches the model tokenizer.
    """
    try:
        tokenization_cfg_path = hf_hub_download(
            dataset_path, "sae_lens.json", repo_type="dataset"
        )
    except HfHubHTTPError:
        return
    if tokenization_cfg_path is None:
        return
    with open(tokenization_cfg_path) as f:
        tokenization_cfg = json.load(f)
    tokenizer_name = tokenization_cfg["tokenizer_name"]
    try:
        ds_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # if we can't download the specified tokenizer to verify, just continue
    except HTTPError:
        return
    if ds_tokenizer.get_vocab() != model_tokenizer.get_vocab():
        raise ValueError(
            f"Dataset tokenizer {tokenizer_name} does not match model tokenizer {model_tokenizer}."
        )


def _get_model_device(model: HookedRootModule) -> torch.device:
    if hasattr(model, "W_E"):
        return model.W_E.device  # type: ignore
    if hasattr(model, "cfg") and hasattr(model.cfg, "device"):
        return model.cfg.device  # type: ignore
    return next(model.parameters()).device  # type: ignore


def _get_special_token_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Get all special token IDs from a tokenizer."""
    special_tokens = set()

    # Get special tokens from tokenizer attributes
    for attr in dir(tokenizer):
        if attr.endswith("_token_id"):
            token_id = getattr(tokenizer, attr)
            if token_id is not None:
                special_tokens.add(token_id)

    # Get any additional special tokens from the tokenizer's special tokens map
    if hasattr(tokenizer, "special_tokens_map"):
        for token in tokenizer.special_tokens_map.values():
            if isinstance(token, str):
                token_id = tokenizer.convert_tokens_to_ids(token)  # type: ignore
                special_tokens.add(token_id)
            elif isinstance(token, list):
                for t in token:
                    token_id = tokenizer.convert_tokens_to_ids(t)  # type: ignore
                    special_tokens.add(token_id)

    return list(special_tokens)


def _filter_buffer_acts(
    buffer: tuple[torch.Tensor, torch.Tensor | None],
    exclude_tokens: torch.Tensor | None,
) -> torch.Tensor:
    """
    Filter out activations for tokens that are in exclude_tokens.
    """

    activations, tokens = buffer
    if tokens is None or exclude_tokens is None:
        return activations

    mask = torch.isin(tokens, exclude_tokens)
    return activations[~mask]


def permute_together(tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Permute tensors together."""
    permutation = torch.randperm(tensors[0].shape[0])
    return tuple(t[permutation] for t in tensors)
