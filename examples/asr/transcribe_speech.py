# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
from dataclasses import dataclass, is_dataclass
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.metrics.rnnt_wer import RNNTDecodingConfig
from nemo.collections.asr.metrics.wer import CTCDecodingConfig
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.parts.utils.transcribe_utils import (
    compute_output_filename,
    prepare_audio_data,
    setup_model,
    transcribe_partial_audio,
    write_transcription,
)
from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.core.config import hydra_runner
from nemo.utils import logging


"""
Transcribe audio file on a single CPU/GPU. Useful for transcription of moderate amounts of audio data.

# Arguments
  model_path: path to .nemo ASR checkpoint
  pretrained_name: name of pretrained ASR model (from NGC registry)
  audio_dir: path to directory with audio files
  dataset_manifest: path to dataset JSON manifest file (in NeMo format)

  compute_langs: Bool to request language ID information (if the model supports it)

  output_filename: Output filename where the transcriptions will be written
  batch_size: batch size during inference

  cuda: Optional int to enable or disable execution of model on certain CUDA device.
  amp: Bool to decide if Automatic Mixed Precision should be used during inference
  audio_type: Str filetype of the audio. Supported = wav, flac, mp3

  overwrite_transcripts: Bool which when set allowes repeated transcriptions to overwrite previous results.

  rnnt_decoding: Decoding sub-config for RNNT. Refer to documentation for specific values.

# Usage
ASR model can be specified by either "model_path" or "pretrained_name".
Data for transcription can be defined with either "audio_dir" or "dataset_manifest".
append_pred - optional. Allows you to add more than one prediction to an existing .json
pred_name_postfix - optional. The name you want to be written for the current model
Results are returned in a JSON manifest file.

python transcribe_speech.py \
    model_path=null \
    pretrained_name=null \
    audio_dir="<remove or path to folder of audio files>" \
    dataset_manifest="<remove or path to manifest>" \
    output_filename="<remove or specify output filename>" \
    batch_size=32 \
    compute_langs=False \
    cuda=0 \
    amp=True \
    append_pred=False \
    pred_name_postfix="<remove or use another model name for output filename>"
"""


@dataclass
class TranscriptionConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest

    # General configs
    output_filename: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 0
    append_pred: bool = False  # Sets mode of work, if True it will add new field transcriptions.
    pred_name_postfix: Optional[str] = None  # If you need to use another model name, rather than standard one.

    # Set to True to output language ID information
    compute_langs: bool = False

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    amp: bool = False
    audio_type: str = "wav"

    # Recompute model transcription, even if the output folder exists with scores.
    overwrite_transcripts: bool = True

    # Decoding strategy for CTC models
    ctc_decoding: CTCDecodingConfig = CTCDecodingConfig()

    # Decoding strategy for RNNT models
    rnnt_decoding: RNNTDecodingConfig = RNNTDecodingConfig(fused_batch_size=-1)


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig) -> TranscriptionConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.dataset_manifest is None:
        raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

    # setup GPU
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
        else:
            device = 1
            accelerator = 'cpu'
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'
    map_location = torch.device('cuda:{}'.format(device[0]) if accelerator == 'gpu' else 'cpu')
    logging.info(f"Inference will be done on device : {device}")

    asr_model, model_name = setup_model(cfg, map_location)

    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    asr_model.set_trainer(trainer)
    asr_model = asr_model.eval()

    # collect additional transcription information
    return_hypotheses = True

    # we will adjust this flag is the model does not support it
    compute_langs = cfg.compute_langs

    # Setup decoding strategy
    if hasattr(asr_model, 'change_decoding_strategy'):
        # Check if ctc or rnnt model
        if hasattr(asr_model, 'joint'):  # RNNT model
            cfg.rnnt_decoding.fused_batch_size = -1
            cfg.rnnt_decoding.compute_langs = cfg.compute_langs
            asr_model.change_decoding_strategy(cfg.rnnt_decoding)
        else:
            asr_model.change_decoding_strategy(cfg.ctc_decoding)

    # prepare audio filepaths and decide wether it's partical audio
    filepaths, partial_audio = prepare_audio_data(cfg)

    # setup AMP (optional)
    if cfg.amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast():
            yield

    # Compute output filename
    cfg = compute_output_filename(cfg, model_name)

    # if transcripts should not be overwritten, and already exists, skip re-transcription step and return
    if not cfg.overwrite_transcripts and os.path.exists(cfg.output_filename):
        logging.info(
            f"Previous transcripts found at {cfg.output_filename}, and flag `overwrite_transcripts`"
            f"is {cfg.overwrite_transcripts}. Returning without re-transcribing text."
        )
        return cfg

    # transcribe audio
    with autocast():
        with torch.no_grad():
            if partial_audio:
                if isinstance(asr_model, EncDecCTCModel):
                    transcriptions = transcribe_partial_audio(
                        asr_model=asr_model,
                        path2manifest=cfg.dataset_manifest,
                        batch_size=cfg.batch_size,
                        num_workers=cfg.num_workers,
                        return_hypotheses=return_hypotheses,
                    )
                else:
                    logging.warning(
                        "RNNT models do not support transcribe partial audio for now. Transcribing full audio."
                    )
                    transcriptions = asr_model.transcribe(
                        paths2audio_files=filepaths,
                        batch_size=cfg.batch_size,
                        num_workers=cfg.num_workers,
                        return_hypotheses=return_hypotheses,
                    )
            else:
                transcriptions = asr_model.transcribe(
                    paths2audio_files=filepaths,
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                    return_hypotheses=return_hypotheses,
                )

    logging.info(f"Finished transcribing {len(filepaths)} files !")
    logging.info(f"Writing transcriptions into file: {cfg.output_filename}")

    # if transcriptions form a tuple (from RNNT), extract just "best" hypothesis
    if type(transcriptions) == tuple and len(transcriptions) == 2:
        transcriptions = transcriptions[0]

    # write audio transcriptions
    output_filename = write_transcription(transcriptions, cfg, model_name, filepaths, compute_langs)
    logging.info(f"Finished writing predictions to {output_filename}!")

    return cfg


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
