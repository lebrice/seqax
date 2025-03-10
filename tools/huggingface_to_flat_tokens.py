"""Tokenizes a Huggingface dataset and writes it to `flat-tokens` format.

See `docs/flat-tokens.md` for details on the format.
See `configs/c4_en.yaml` for an instructions on running.

TODO: we could make this much faster by sharding over multiple CPUs. Rough approach:
1) Make this script read from a shard of the Huggingface dataset.
2) At the end of this script, wait for all shards to complete, and then concatenate the zarr data.
"""

import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

import tqdm

import flat_tokens
import hydra
import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from logging import getLogger
import rich.logging

logger = getLogger(__name__)


@dataclass(frozen=True)
class Config:
    output: str
    tokenizer: str
    dataset: str
    variant: Optional[str]
    max_tokens: Optional[int]
    write_buffer_size_in_sequences: int
    flat_tokens_config: flat_tokens.Config
    _target_: str = __name__ + ".Config"


@dataclass(frozen=True)
class FlatTokensConfig:
    tokens_chunk_size: int = 4194304
    seq_starts_chunk_size: int = 65536


@dataclass(frozen=True)
class SyntheticDatasetConfig(Config):
    seed: int = 1
    seq_len: int = 1024
    examples: int = 512
    flat_tokens_config: FlatTokensConfig = FlatTokensConfig()


# Registering the Config class with the name 'config'.
# ConfigStore.instance().store(name="config_schema", node=Config)
# ConfigStore.instance().store(
#     name="synthetic_dataset_schema", node=SyntheticDatasetConfig
# )


def setup_logging(local_rank: int, num_processes: int, verbose: int):
    logging.basicConfig(
        level=logging.INFO,
        # Add the [{local_rank}/{num_processes}] prefix to log messages
        format=f"[%(asctime)s][%(levelname)s][{local_rank + 1}/{num_processes}] %(message)s",
        handlers=[rich.logging.RichHandler()],
    )
    if verbose == 0:
        logger.setLevel(logging.ERROR)
    elif verbose == 1:
        logger.setLevel(logging.WARNING)
    elif verbose == 2:
        logger.setLevel(logging.INFO)
    else:
        assert verbose >= 3
        logger.setLevel(logging.DEBUG)


def _tokenize_and_concat(batch, tokenize: Callable):
    chunk = flat_tokens.Chunk.from_ragged(tokenize(batch["text"]))
    # dataset.map() API requires us to return numpy tensors of the appropriate shape...
    return {
        "encoded_tokens": chunk.encoded_tokens[np.newaxis, :],
        "seq_starts": chunk.seq_starts[np.newaxis, :],
        "max_token_id": np.array(chunk.max_token_id, np.uint32)[np.newaxis],
    }


@hydra.main(config_path="configs", version_base=None)
def main(config):
    logger.info(config)
    # Create tokenizer

    setup_logging(0, 1, 3)

    if config.tokenizer == "bytes_utf8":

        def tokenize(texts):
            return [
                np.uint32(np.frombuffer(text.encode("utf-8"), np.uint8)) + 1
                for text in texts
            ]
    else:
        logger.info(f"Loading tokenizer {config.tokenizer}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer)
        assert 0 in tokenizer.all_special_ids, "Tokenizer must have 0 as a special id"
        assert tokenizer.vocab_size < 1 << 31, (
            "Tokenizer vocab size too large for uint31"
        )

        def tokenize(texts):
            return tokenizer(texts, add_special_tokens=False)["input_ids"]

    tokenize_and_concat = functools.partial(_tokenize_and_concat, tokenize=tokenize)

    executor = ThreadPoolExecutor()

    for split, mode in [("validation", "w-"), ("train", "r+")]:
        logger.info(f"Processing split '{split}'")
        logger.info(f"Loading dataset {config.dataset}")
        dataset = load_dataset(
            config.dataset,
            config.variant,
            # streaming=True,
            split=split,
        )
        dataset = dataset.select_columns(["text"])
        dataset = dataset.map(
            tokenize_and_concat,
            batched=True,
            batch_size=config.write_buffer_size_in_sequences,
            remove_columns=["text"],
        )

        # Open output
        dst = flat_tokens.Writer(
            config.output, flat_tokens.Split(split), mode, config.flat_tokens_config
        )
        dst_flush = executor.submit(lambda: None)

        # Write in batches
        flush_elapsed = 0
        start_time = time.time()
        next_update = 0
        seq_count = 0
        token_count = 0
        for batch in tqdm.tqdm(dataset):
            chunk = flat_tokens.Chunk(
                encoded_tokens=batch["encoded_tokens"],
                seq_starts=batch["seq_starts"],
                max_token_id=batch["max_token_id"],
            )
            seq_count += len(chunk.seq_starts) - 1
            token_count += len(chunk.encoded_tokens)

            flush_start = time.time()
            dst_flush.result()
            dst_flush = executor.submit(dst.write, chunk)
            flush_elapsed += time.time() - flush_start
            elapsed = time.time() - start_time
            if elapsed > next_update:
                total_mib = token_count * 4 // (1024 * 1024)
                speed_mib_per_s = total_mib / elapsed
                print(
                    f"[{int(elapsed):_}s] Processed {seq_count:_} examples, {token_count:_} tokens, {total_mib:_} MiB, {speed_mib_per_s:.2f} MiB/s. Flush time: {flush_elapsed:.2f}s"
                )
                next_update = elapsed + 60

            if token_count >= config.max_tokens:
                break

        # Final flush
        dst_flush.result()

        print(
            f"Done with split '{split}': {seq_count:_} examples, {token_count:_} tokens"
        )


if __name__ == "__main__":
    main()
