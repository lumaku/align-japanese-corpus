#!/usr/bin/env python3
# Copyright 2021, Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Evaluate ground truth sequences by alignment.
Either start this program as a script or from the interactive python REPL.
Example:
ngpu = 1
dataset_path = Path("/zzz/20210304/")
dir_wav =  dataset_path / "watanabe-sensei_pilot-data" / "wav16k"
dir_txt =  dataset_path / "watanabe-sensei_pilot-data" / "txt"
output = dataset_path
re_segmentation = True
SKIP_LONG_FILES_S = 492.0
utterance_scoring(log_level="INFO", wavdir=dir_wav, txtdir=dir_txt, output=output, ngpu=ngpu, re_segmentation=re_segmentation)
"""

import argparse
import logging
import sys
from typing import Union

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError

from espnet.utils.cli_utils import get_commandline_args
from espnet2.utils import config_argparse
from espnet2.utils.types import str_or_none

from pathlib import Path
import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_align import CTCSegmentation
from torch.multiprocessing import Process, Queue

# Language specific imports - japanese
import romkan
from num2words import num2words
import re

NUMBER_OF_PROCESSES = 4
SKIP_LONG_FILES_S = 1000000000.0


def align_worker(in_queue, out_queue, num=0):
    print(f"align_worker {num} started")
    for task, meta_data in iter(in_queue.get, "STOP"):
        try:
            result = CTCSegmentation.get_segments(task)
            task.set(**result)
            if meta_data["scoring"]:
                s, e, score = task.segments[0]
                task.segments[0] = (meta_data["start"], meta_data["end"], score)
            segments_str = str(task)
            out_queue.put(segments_str)
        except (AssertionError, IndexError) as e:
            # AssertionError: Audio is shorter than ground truth
            # IndexError: backtracking not successful
            logging.error(
                f"Failed to align {task.utt_ids[0]} in {task.name} because of: {e}"
            )
        del task
    print(f"align_worker {num} stopped")


def listen_worker(in_queue, segments="./segments.txt"):
    print("listen_worker started.")
    with open(segments, "w") as f:
        for item in iter(in_queue.get, "STOP"):
            if segments is None:
                print(item)
            else:
                f.write(item)
                f.flush()
    print("listen_worker ended.")


def main_re_segmentation(
    task_queue,
    aligner,
    wav,
    utterance_list,
    stem,
    count_files,
    num_files,
    fs,
    done_queue,
):
    text = []
    meta_data = {"scoring": False}
    for i, utt in enumerate(utterance_list):
        utt_start, utt_end, utt_txt = utt.split(" ", 2)

        # text processing
        utt_txt = text_processing(utt_txt)
        cleaned = aligner.preprocess_fn.text_cleaner(utt_txt)
        text.append(f"{stem}_{i:04} {cleaned}")

    speech, sample_rate = soundfile.read(wav)
    assert fs == sample_rate
    duration = speech.shape[0] / sample_rate
    logging.info(
        f"Inference on file {count_files}/{num_files}: {len(utterance_list)} utterances:  ({duration}s)"
    )
    if SKIP_LONG_FILES_S < duration:
        logging.info(f"SKIPPED {stem} for length.")
        return
    task = None
    try:
        lpz = aligner.get_lpz(speech)
        task = aligner.prepare_segmentation_task(
            text, lpz, name=stem, speech_len=speech.shape[0]
        )
        task.name = stem
    except Exception as e:
        # RuntimeError: unknown CUDA value error (at inference)
        # TooShortUttError: Audio too short (at inference)
        # IndexError:ground: truth is empty (thrown at preparation)
        logging.error(f"LPZ failed for file {stem}; error in espnet: {e}")
    try:
        if task:
            logging.info(
                f"Aligning file {count_files}/{num_files}: {len(utterance_list)} utterances: {stem}"
            )
            result = CTCSegmentation.get_segments(task)
            task.set(**result)
            segments_str = str(task)
            done_queue.put(segments_str)
    except (AssertionError, IndexError) as e:
        # AssertionError: Audio is shorter than ground truth
        # IndexError: backtracking not successful
        logging.error(
            f"Failed to align {task.utt_ids[0]} in {task.name} because of: {e}"
        )
    del task


def main_scoring(
    task_queue, aligner, wav, utterance_list, stem, count_files, num_files, fs
):
    logging.info(
        f"Scoring file {count_files}/{num_files}: {len(utterance_list)} utterances: {stem}"
    )
    for i, utt in enumerate(utterance_list):
        utt_start, utt_end, utt_txt = utt.split(" ", 2)
        # start and end are not aligned, but fixed, and need to given as separate info.
        utt_start = float(utt_start)
        utt_end = float(utt_end)
        meta_data = {"start": utt_start, "end": utt_end, "scoring": True}
        utt_start = int(utt_start * fs)
        utt_end = int(utt_end * fs)

        # text processing
        utt_txt = text_processing(utt_txt)
        cleaned = aligner.preprocess_fn.text_cleaner(utt_txt)
        text = f"{stem}_{i:04} {cleaned}"

        speech, sample_rate = soundfile.read(wav, start=utt_start, stop=utt_end)
        assert fs == sample_rate
        try:
            lpz = aligner.get_lpz(speech)
            task = aligner.prepare_segmentation_task(
                text, lpz, name=stem, speech_len=speech.shape[0]
            )
            task.name = stem
            task_queue.put(
                (
                    task,
                    meta_data,
                )
            )
        except (RuntimeError, TooShortUttError, IndexError) as e:
            # RuntimeError: unknown CUDA value error (at inference)
            # TooShortUttError: Audio too short (at inference)
            # IndexError:ground: truth is empty (thrown at preparation)
            logging.error(
                f"LPZ failed for utterance {stem}_{i:04} ({meta_data}); error in espnet: {e}"
            )


def text_processing(utt_txt):
    # text processing
    # normalize text
    utt_txt = utt_txt.replace('"', "").replace(",", "")
    utt_txt = romkan.to_hiragana(utt_txt)
    # replace all the numbers
    numbers = re.findall(r"\d+", utt_txt)
    transcribed_numbers = [num2words(item, lang="ja") for item in numbers]
    for nr in range(len(numbers)):
        old_nr = numbers[nr]
        new_nr = transcribed_numbers[nr]
        utt_txt = utt_txt.replace(old_nr, new_nr, 1)
    return utt_txt


# One remark about the "experimental" option of utterance_scoring: I modified the
# CTC segmentation algorithm so that all frames from start are counted at the
# start and at the ending of the utterance (usually, preamble and ending is ignored)
def utterance_scoring(
    log_level: Union[int, str],
    wavdir: Path,
    txtdir: Path,
    output: Path,
    experimental: bool = False,
    re_segmentation: bool = False,
    **kwargs,
):
    """Provide the scripting interface to score text to audio."""
    assert check_argument_types()
    # make sure that output is a path!
    logfile = output / "segments.log"
    segments = output / "segments.txt"
    logging.basicConfig(
        level=log_level,
        filename=str(logfile),
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # Ignore configuration values that are set to None (from parser).
    kwargs = {k: v for (k, v) in kwargs.items() if v is not None}

    # Prepare CTC segmentation module
    asr_model_name = "Shinji Watanabe/laborotv_asr_train_asr_conformer2_latest33_raw_char_sp_valid.acc.ave"
    d = ModelDownloader(cachedir="./modelcache")
    model = d.download_and_unpack(asr_model_name)
    aligner = CTCSegmentation(**model, **kwargs, kaldi_style_text=True)
    fs = 16000

    ## application-specific settings
    # japanese text cleaning
    aligner.preprocess_fn.text_cleaner.cleaner_types += ["jaconv"]
    # ensure that min window size includes max audio length
    max_length_audio_s = 25
    indizes_per_second = 32
    aligner.config.min_window_size = max(8000, indizes_per_second * max_length_audio_s)
    # preamble and postamble shall not have zero transition cost
    if experimental:
        aligner.config.preamble_transition_cost_zero = False
        aligner.config.backtrack_from_max_t = True
    # set length of scoring avgerage window (1s ~ 32)
    aligner.config.score_min_mean_over_L = 30

    # Create queues
    task_queue = Queue(maxsize=NUMBER_OF_PROCESSES)
    done_queue = Queue()

    files_dict = {}
    dir_txt_list = list(txtdir.glob("**/*.txt"))
    for wav in wavdir.glob("**/*.wav"):
        stem = wav.stem
        txt = None
        for item in dir_txt_list:
            if item.stem == stem:
                if txt is not None:
                    raise ValueError(f"Duplicate found: {stem}")
                txt = item
        if txt is None:
            logging.error(f"No text found for {stem}.wav")
        else:
            files_dict[stem] = (wav, txt)
    num_files = len(files_dict)
    logging.info(f"Found {num_files} files.")


    # Start worker processes
    Process(
        target=listen_worker,
        args=(
            done_queue,
            segments,
        ),
    ).start()
    for i in range(NUMBER_OF_PROCESSES):
        Process(target=align_worker, args=(task_queue, done_queue, i)).start()

    count_files = 0
    for stem in files_dict.keys():
        count_files += 1
        (wav, txt) = files_dict[stem]
        with open(txt) as f:
            utterance_list = f.readlines()
        utterance_list = [
            item.replace("\t", " ").replace("\n", "") for item in utterance_list
        ]
        if re_segmentation:
            # re-align all utterances
            main_re_segmentation(
                task_queue,
                aligner,
                wav,
                utterance_list,
                stem,
                count_files,
                num_files,
                fs,
                done_queue,
            )
        else:
            # re-score all utterances
            main_scoring(
                task_queue,
                aligner,
                wav,
                utterance_list,
                stem,
                count_files,
                num_files,
                fs,
            )

    logging.info("Shutting down workers.")
    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")
    done_queue.put("STOP")


def get_parser():
    """Obtain an argument-parser for the script interface."""
    parser = config_argparse.ArgumentParser(
        description="ASR Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ASR model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )

    group = parser.add_argument_group("CTC segmentation related")
    group.add_argument(
        "--fs",
        type=int,
        default=16000,
        help="Sampling Frequency."
        " The sampling frequency (in Hz) is needed to correctly determine the"
        " starting and ending time of aligned segments.",
    )

    group = parser.add_argument_group("Input/output arguments")
    group.add_argument(
        "--wavdir",
        type=Path,
        required=True,
        help="WAV folder.",
    )
    group.add_argument(
        "--txtdir",
        type=Path,
        required=True,
        help="Text files folder.",
    )
    group.add_argument(
        "--output",
        type=Path,
        help="Output segments directory.",
    )
    return parser


def main(cmd=None):
    """Parse arguments and start."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    utterance_scoring(**kwargs)


if __name__ == "__main__":
    main()