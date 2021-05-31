#!/usr/bin/env python3

# Copyright 2021, Ludwig Kürzinger, Takaaki Saeki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Perform CTC Re-Segmentation on japanese dataset.
Either start this program as a script or from the interactive python REPL.
Example:
ngpu = 0     # recommended for long audio files
## or:
LONGEST_AUDIO_SEGMENTS = 75
ngpu = 1
dataset_path = Path("/zzz/20210304/")
dir_wav =  dataset_path / "watanabe-sensei_pilot-data" / "wav16k"
dir_txt =  dataset_path / "watanabe-sensei_pilot-data" / "txt"
output = dataset_path
## Choose a Model
# Japanese RNN by Shinji
model_path = dataset_path / "exp/asr_train_asr_rnn_raw_jp_char_sp"
asr_model_file = model_path / "7epoch.pth"
asr_train_config = model_path / "config.yaml"
# English Librispeech Sinc-RNN by Ludwig
(to test inference speed); commit 473128be9f9c358b3564a2aaff8b7056b790cd12
model_path = dataset_path / "exp/asr_train_asr_sinc_rnn_raw_bpe5000_sp"
asr_model_file = model_path / "9epoch.pth"
asr_train_config = model_path / "config.yaml"
model = {
    "asr_train_config": asr_train_config,
    "asr_model_file": asr_model_file,
}
# Japanese Transformer Model by Shinji
asr_model_name = "Shinji Watanabe/laborotv_asr_train_asr_conformer2_latest33_raw_char_sp_valid.acc.ave"
d = ModelDownloader(cachedir="./modelcache")
model = d.download_and_unpack(asr_model_name)
# note: this model has FRAMES_PER_INDEX=768
align(log_level="INFO", wavdir=dir_wav, txtdir=dir_txt, output=output, ngpu=ngpu, gratis_blank=True, **model)
"""

import argparse
import logging
import sys
import time
from typing import Union
import torch

from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.utils import config_argparse
from espnet2.utils.types import str_or_none

from pathlib import Path
import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_align import CTCSegmentation
from torch.multiprocessing import Process, Queue
from espnet2.utils.types import str2bool

# Language specific imports - japanese
from num2words import num2words
import re

try:
    import neologdn

    NEOLOGDN_AVAILABLE = True
except:
    print("ERROR: neologdn is not available!")
    NEOLOGDN_AVAILABLE = False
try:
    import romkan

    ROMKAN_AVAILABLE = True
except:
    print("ERROR: romkan is not available!")
    ROMKAN_AVAILABLE = False

LONGEST_AUDIO_SEGMENTS = 320
# NUMBER_OF_PROCESSES determines how many CTC segmentation workers
# are started. Set this higher or lower, depending how fast your
# network can do the inference and how much RAM you have
NUMBER_OF_PROCESSES = 4


def align_worker(in_queue, out_queue, num=0):
    print(f"align_worker {num} started")
    for task in iter(in_queue.get, "STOP"):
        try:
            result = CTCSegmentation.get_segments(task)
            task.set(**result)
            segments_str = str(task)
            out_queue.put(segments_str)
            # calculate average score
            scores = [boundary[2] for boundary in task.segments]
            avg = sum(scores) / len(scores)
            logging.info(f"Aligned {task.name} with avg score {avg:3.4f}")
        except (AssertionError, IndexError) as e:
            # AssertionError: Audio is shorter than ground truth
            # IndexError: backtracking not successful (e.g. audio-text mismatch)
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


def get_partitions(
    t=100000,
    max_len_s=1280.0,
    fs=16000,
    frontend_frame_size=512,
    subsampling_factor=4,
):
    """Obtain partitions
    Note that this is implemented for frontends that discard trailing data.
    :param t: speech sample points
    :param max_len_s: partition max length in seconds
    :param fs: sample rate
    :param frontend_frame_size: usually 512
    :param subsampling_factor: subsampling factor, usually 4
    :return:
    """
    # max length should be ~ cut length + 25%
    cut_time_s = max_len_s / 1.25
    max_length = int(max_len_s * fs)
    cut_length = int(cut_time_s * fs)
    samples_per_ctc_index = int(frontend_frame_size * subsampling_factor)
    # make sure its a multiple of frame size
    max_length -= max_length % samples_per_ctc_index
    cut_length -= max_length % samples_per_ctc_index
    assert type(max_length) == int
    assert type(cut_length) == int
    if (max_length - cut_length) <= samples_per_ctc_index:
        raise ValueError("Pick a larger time value for partitions.")
    partitions = []
    start = 0
    while t > max_length:
        t -= cut_length
        end = start + cut_length + frontend_frame_size
        partitions += [(start, end)]
        start += cut_length
    else:
        partitions += [(start, None)]
    return partitions


def text_processing(utt_txt):
    """Normalize text
    Use for Japanese text.
    Args:
        utt_txt: String of Japanese text.
    Returns:
        utt_txt: Normalized
    """
    # convert UTF-16 latin chars to ASCII
    if NEOLOGDN_AVAILABLE:
        utt_txt = neologdn.normalize(utt_txt)
    # Romanji to Hiragana
    if ROMKAN_AVAILABLE:
        utt_txt = romkan.to_hiragana(utt_txt)
    # replace some special characters
    utt_txt = utt_txt.replace('"', "").replace(",", "")
    # replace all the numbers
    numbers = re.findall(r"\d+\.?\d*", utt_txt)
    transcribed_numbers = [num2words(item, lang="ja") for item in numbers]
    for nr in range(len(numbers)):
        old_nr = numbers[nr]
        new_nr = transcribed_numbers[nr]
        utt_txt = utt_txt.replace(old_nr, new_nr, 1)
    return utt_txt


def align(
    log_level: Union[int, str],
    wavdir: Path,
    txtdir: Path,
    output: Path,
    asr_train_config: Union[Path, str],
    asr_model_file: Union[Path, str] = None,
    fixed_time_stamps=False,
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
    model = {
        "asr_train_config": asr_train_config,
        "asr_model_file": asr_model_file,
    }
    logging.info(f"Loading ASR model from {asr_model_file}")
    aligner = CTCSegmentation(**model, **kwargs, kaldi_style_text=True)
    fs = 16000
    logging.info(
        f"Zero cost transitions (gratis_blank) set to"
        f" {aligner.config.blank_transition_cost_zero}."
    )

    # fixed ratio for timing stamps
    if fixed_time_stamps:
        aligner.set_config(
            time_stamps="fixed",
        )
    # estimated index to frames ratio, usually 512, but sometimes 768
    # - depends on architecture
    aligner.set_config(
        samples_to_frames_ratio=aligner.estimate_samples_to_frames_ratio(),
    )
    logging.info(
        f"Timing ratio (sample points per CTC index) set to"
        f" {aligner.samples_to_frames_ratio} ({aligner.time_stamps})."
    )

    ## application-specific settings
    # japanese text cleaning
    aligner.preprocess_fn.text_cleaner.cleaner_types += ["jaconv"]

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

    # Align
    count_files = 0
    for stem in files_dict.keys():
        count_files += 1
        (wav, txt) = files_dict[stem]

        # generate kaldi-style `text`
        with open(txt) as f:
            utterance_list = f.readlines()
        utterance_list = [
            item.replace("\t", " ").replace("\n", "") for item in utterance_list
        ]
        text = []
        for i, utt in enumerate(utterance_list):
            utt_start, utt_end, utt_txt = utt.split(" ", 2)
            # text processing
            utt_txt = text_processing(utt_txt)
            cleaned = aligner.preprocess_fn.text_cleaner(utt_txt)
            text.append(f"{stem}_{i:04} {cleaned}")

        # audio
        speech, sample_rate = soundfile.read(wav)
        speech_len = speech.shape[0]
        speech = torch.tensor(speech)
        partitions = get_partitions(
            speech_len,
            max_len_s=LONGEST_AUDIO_SEGMENTS,
            frontend_frame_size=aligner.samples_to_frames_ratio,
        )
        duration = speech_len / sample_rate
        expected_lpz_length = speech_len // aligner.samples_to_frames_ratio

        logging.info(
            f"Inference on file {stem} {count_files}/{num_files}: {len(utterance_list)}"
            f" utterances:  ({duration}s ~{len(partitions)}p)"
        )
        try:
            # infer
            lpzs = [
                torch.tensor(aligner.get_lpz(speech[start:end]))
                for start, end in partitions
            ]
            lpz = torch.cat(lpzs).numpy()
            if not lpz.shape[0] == expected_lpz_length:
                logging.error(
                    f"LPZ size mismatch on {stem}: "
                    f"{lpz.shape[0]}-{expected_lpz_length}"
                )
            task = aligner.prepare_segmentation_task(
                text, lpz, name=stem, speech_len=speech_len
            )
            # align (done by worker)
            task_queue.put(task)
        except KeyboardInterrupt:
            break
        except Exception as e:
            # RuntimeError: unknown CUDA value error (at inference)
            # TooShortUttError: Audio too short (at inference)
            # IndexError: ground truth is empty (thrown at preparation)
            # Assertion Error for audio-lpz length mismatch
            logging.error(f"LPZ failed for file {stem}; error in espnet: {e}")
    logging.info("Shutting down workers.")
    # wait for workers to finish
    time.sleep(5)
    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")
    done_queue.put("STOP")


def get_parser():
    """Obtain an argument-parser for the script interface."""
    parser = config_argparse.ArgumentParser(
        description="CTC segmentation",
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
    group.add_argument(
        "--gratis_blank",
        type=str2bool,
        default=True,
        help="Set the transition cost of the blank token to zero. Audio sections"
        " labeled with blank tokens can then be skipped without penalty. Useful"
        " if there are unrelated audio segments between utterances.",
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--asr_train_config", type=str, required=True)
    group.add_argument("--asr_model_file", type=str, required=True)

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
    align(**kwargs)


if __name__ == "__main__":
    main()