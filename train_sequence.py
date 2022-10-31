"""

    Train Sequence Model

"""
from datetime import datetime
from itertools import chain
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional

import fire
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from melreformer.core.sequence.trainer import SequenceTrainer
from melreformer.data.core.chunk import AudioChunkDataset, StableAudioChunkDataset
from melreformer.data.core.sized import SizedDataset
from melreformer.utils.pl_callbacks import CustomTqdmProgressBar
from melreformer.utils.reproducibility import SEED_VALUE, seed_everything

START_TIME = datetime.utcnow()

seed_everything(deterministic_cudnn=False)
DEFAULT_DATASET_CACHE_DIR = str(Path("~/datasets/cache").expanduser())


def _make_datasets(
    data_paths: list[str],
    n_segments: int,
    segment_length: int,
    sample_rate: int = 22_050,
    test_size: float = 0.1,
    ext: str = "wav",
    train_samples_per_epoch: int = 2**15,
    dataset_cache_dir: str = DEFAULT_DATASET_CACHE_DIR,
    verbose: bool = True,
) -> tuple[SizedDataset, StableAudioChunkDataset]:
    duration: float = (segment_length * n_segments) / sample_rate

    all_files = list(chain(*[Path(p).rglob(f"*.{ext}") for p in data_paths]))
    if verbose:
        print(f"Found {len(all_files)} audio files in dataset.")

    train_files, val_files = train_test_split(
        all_files,
        test_size=test_size,
        random_state=SEED_VALUE,
    )
    train_dataset = SizedDataset(
        AudioChunkDataset(
            src_files=train_files,
            cache_dir=dataset_cache_dir,
            sr=sample_rate,
            duration=duration,
        ),
        size=train_samples_per_epoch,
    )
    val_dataset = StableAudioChunkDataset(
        src_files=val_files,
        cache_dir=dataset_cache_dir,
        sr=sample_rate,
        duration=duration,
        size_override=4096,
    )
    return train_dataset, val_dataset


def main(
    data_paths: str,
    vocoder_path: str,
    # Output & Cache
    base_dir: Optional[str] = None,
    dataset_cache_dir: str = DEFAULT_DATASET_CACHE_DIR,
    # W&B
    wb_name: Optional[str] = None,
    project_name: Optional[str] = None,
    # Audio
    n_segments: int = 64,
    segment_length: int = 8192,
    # Data
    batch_size: int = 32,
    train_batch_ceil: Optional[int] = 2**10,
    **kwargs,
) -> None:
    """Train audio segment model.

    Args:
        data_paths (str): a comma-separated list of system paths to the
            training dataset(s)
        vocoder_path (str): a system path to a pretrained HifiVocoder
        base_dir (str):  a system path to use the "base" or reference
            when writing output. If None, defaults to the home directory.
        dataset_cache_dir (str): system path to cache loaded audio files to
        wb_name (str, optional): username for W&B
        project_name (str): name of the project to use when writing results to
            weights and biases. Will be ignored if `wb_name` is None.
        n_segments (int): number of audio segments to model
        segment_length (int): number of audio samples in each segment
        batch_size (int): number of audio segments in each batch
        train_batch_ceil (int): maximum number of audio segments to use
        **kwargs (Keyword Args): Keyword arguments to control construction
            of audio datasets

    Returns:
        None

    """
    train_dataset, val_dataset = _make_datasets(
        data_paths=data_paths.split(","),
        n_segments=n_segments,
        segment_length=segment_length,
        dataset_cache_dir=dataset_cache_dir,
        **kwargs,
    )

    model = SequenceTrainer(
        vocoder_path=vocoder_path,
        n_segment=n_segments,
        start_time=START_TIME,
        base_dir=base_dir or str(Path("~/").expanduser()),
    )

    Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=-1,
        precision=32,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        limit_train_batches=train_batch_ceil,
        log_every_n_steps=5,
        logger=(
            WandbLogger(
                project=project_name,
                name=START_TIME,
                entity=wb_name,
                anonymous=False,
                # see https://docs.wandb.ai/guides/track/launch#init-start-error
                settings=wandb.Settings(start_method="fork"),
            )
            if wb_name
            else None
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=str(model.checkpoints_dir),
                filename="{epoch:02d}-{g_loss_val}",
                monitor="g_loss_val",
                mode="min",
                save_top_k=3,
                save_last=True,
            ),
            CustomTqdmProgressBar(),
        ],
    ).fit(
        model,
        train_dataloaders=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        ),
        val_dataloaders=DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(SEED_VALUE),
        ),
    )


if __name__ == "__main__":
    fire.Fire(main)
