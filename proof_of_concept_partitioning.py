## Paritioning proof of concept

import torch
import logging
from espnet_model_zoo.downloader import ModelDownloader
import soundfile
from pathlib import Path
from espnet2.bin.asr_align import CTCSegmentation

# load model
logging.basicConfig(
    level="INFO",
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


def get_partitions_split(
    t=100000,
    max_len_s=1280.0,
    fs=16000,
    frontend_frame_size=512,
):
    """Partition t into parts with max length.

    Compatible with torch.split.
    But will generate wrong number of lpz indices.
    Please ignore.
    """
    # max length should be ~ cut length + 25%
    cut_time_s = max_len_s / 1.25
    max_length = int(max_len_s * fs)
    cut_length = int(cut_time_s * fs)
    # make sure its a multiple of frame size
    max_length -= max_length % frontend_frame_size
    cut_length -= max_length % frontend_frame_size
    assert type(max_length) == int
    assert type(cut_length) == int
    partitions = []
    while t > max_length:
        t -= cut_length
        partitions += [cut_length]
    else:
        partitions += [t]
    return partitions


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


## CONFIGURATION
wav = Path("./test_utils/ctc_align_test.wav")
LONGEST_AUDIO_SEGMENTS = 3

text = """
utt1 THE SALE OF THE HOTELS
utt2 IS PART OF HOLIDAY'S STRATEGY
utt3 TO SELL OFF ASSETS
utt4 AND CONCENTRATE ON PROPERTY MANAGEMENT
"""


d = ModelDownloader(cachedir="./modelcache")
wsjmodel = d.download_and_unpack("kamo-naoyuki/wsj")
speech, rate = soundfile.read(wav)

# "segments" without splitting
aligner = CTCSegmentation(**wsjmodel, fs=rate)
segments = aligner(speech, text)
print("regular:")
print(segments)

# "task" with splitted audio
name = "test"
speech_len = speech.shape[0]
speech = torch.tensor(speech)
partitions = get_partitions(speech.shape[0], max_len_s=LONGEST_AUDIO_SEGMENTS)
lpzs = [torch.tensor(aligner.get_lpz(speech[start:end])) for start, end in partitions]
lpz = torch.cat(lpzs).numpy()
task = aligner.prepare_segmentation_task(text, lpz, name, speech_len)
# Apply CTC segmentation
task.set(**CTCSegmentation.get_segments(task))
print(f"splitted {LONGEST_AUDIO_SEGMENTS}:")
print(task)

# "taskmin" with min t for splitting
partitions = get_partitions(speech.shape[0], max_len_s=1.05)
lpzs = [torch.tensor(aligner.get_lpz(speech[start:end])) for start, end in partitions]
lpz = torch.cat(lpzs).numpy()
taskmin = aligner.prepare_segmentation_task(text, lpz, name, speech_len)
# Apply CTC segmentation
taskmin.set(**CTCSegmentation.get_segments(task))
print("splitted min. t :")
print(taskmin)

# make sure output has the same length
assert segments.lpz.shape[0] == task.lpz.shape[0]

# plot activations
# outdir = Path("../align-japanese-corpus/proof_of_concept_partitioning/")
# [plot_activations(outdir, i, aligner.config.char_list[i]) for i in range(len(aligner.config.char_list))]
def plot_activations(outdir: Path, index=2, name=""):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(rc={"figure.figsize": (11.7, 8.27)})
    sns.set_style("white")
    activations = {
        "regular": segments.lpz[:, index],
        f"splitted_{LONGEST_AUDIO_SEGMENTS}s": task.lpz[:, index],
        "splitted_1s": taskmin.lpz[:, index],
    }
    sns.lineplot(data=activations)
    filename = f"plot_{index}.png"
    print("Plot: ", filename)
    plt.ylim(-25, 0)
    plt.title(f"Activations for >{name}<")
    plt.savefig(outdir / filename)
    plt.clf()
