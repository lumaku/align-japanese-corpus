## Paritioning proof of concept
import numpy as np
import torch
import logging
from espnet_model_zoo.downloader import ModelDownloader
import soundfile
from pathlib import Path
from espnet2.bin.asr_align import CTCSegmentation

# from files in the same directory
# make sure these files are in your PYTHONPATH
from align import get_partitions

# load model
logging.basicConfig(
    level="INFO",
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


## CONFIGURATION
wav = Path("../espnet/test_utils/ctc_align_test.wav")
LONGEST_AUDIO_SEGMENTS = 3

text = (
    "THE SALE OF THE HOTELS IS PART OF HOLIDAY'S"
    " STRATEGY TO SELL OFF ASSETS AND CONCENTRATE"
    " ON PROPERTY MANAGEMENT".split()
)

d = ModelDownloader(cachedir="./modelcache")
model = d.download_and_unpack("kamo-naoyuki/wsj")

speech, rate = soundfile.read(wav)

aligner = CTCSegmentation(**model, fs=rate, time_stamps="fixed", kaldi_style_text=False)
aligner.set_config(samples_to_frames_ratio=aligner.estimate_samples_to_frames_ratio())
# estimated index to frames ratio, usually 512, but sometimes 768
# - depends on architecture
logging.info(
    f"Timing ratio (sample points per CTC index) set to"
    f" {aligner.samples_to_frames_ratio}."
)
# "segments" without splitting
segments = aligner(speech, text)
print("regular:")
print(segments)

#####  Task uses audio that is splitted in 3s parts
# "task" with splitted audio
name = "test"
speech_len = speech.shape[0]
speech = torch.tensor(speech)
partitions = get_partitions(speech.shape[0], max_len_s=LONGEST_AUDIO_SEGMENTS)[
    "partitions"
]
lpzs = [torch.tensor(aligner.get_lpz(speech[start:end])) for start, end in partitions]
lpz = torch.cat(lpzs).numpy()
task = aligner.prepare_segmentation_task(text, lpz, name, speech_len)
# Apply CTC segmentation
task.set(**CTCSegmentation.get_segments(task))
print(f"splitted into {LONGEST_AUDIO_SEGMENTS}s:")
print(task)

# make sure output has the same length
if not segments.lpz.shape[0] == task.lpz.shape[0]:
    print(f"ERROR: exp. {segments.lpz.shape[0]} != got {task.lpz.shape[0]} (task)")


#####  Task_min uses audio that is splitted in 1s parts (shortest possible length for splitting)
# "task_min" with min t for splitting
partitions = get_partitions(speech.shape[0], max_len_s=1.05)["partitions"]
lpzs = [torch.tensor(aligner.get_lpz(speech[start:end])) for start, end in partitions]
lpz = torch.cat(lpzs).numpy()
task_min = aligner.prepare_segmentation_task(text, lpz, name, speech_len)
# Apply CTC segmentation
task_min.set(**CTCSegmentation.get_segments(task_min))
print("splitted into smallest parts (min. t ~1s):")
print(task_min)

# make sure output has the same length
if not segments.lpz.shape[0] == task_min.lpz.shape[0]:
    print(
        f"ERROR: exp. {segments.lpz.shape[0]} != got {task_min.lpz.shape[0]} (task_min)"
    )

###### task_overlap is also split into 3s parts but additionally includes a little bit more audio
###### on the intersections between parts; also, the excess lpz indices are deleted.
partitions = get_partitions(
    speech.shape[0], max_len_s=LONGEST_AUDIO_SEGMENTS, overlap=1
)
lpzs = [
    torch.tensor(aligner.get_lpz(speech[start:end]))
    for start, end in partitions["partitions"]
]
lpz = torch.cat(lpzs).numpy()
lpz = np.delete(lpz, partitions["delete_overlap_list"], axis=0)
task_overlap = aligner.prepare_segmentation_task(text, lpz, name, speech_len)
# Apply CTC segmentation
task_overlap.set(**CTCSegmentation.get_segments(task_overlap))
print("With 1 overlapping lpz index:")
print(task_overlap)

# make sure output has the same length
if not segments.lpz.shape[0] == task_overlap.lpz.shape[0]:
    print(
        f"ERROR: exp. {segments.lpz.shape[0]} != got {task_overlap.lpz.shape[0]} (task_overlap)"
    )


###### task_overlap_n is the same as task_overlap, but now has N additional lpz indices
OVERLAP = 10
print(
    f"Partitioning over {LONGEST_AUDIO_SEGMENTS}s."
    f" Overlap time: {512/16000*(2*OVERLAP)}s (overlap={OVERLAP})"
)
partitions = get_partitions(
    speech.shape[0], max_len_s=LONGEST_AUDIO_SEGMENTS, overlap=OVERLAP
)
lpzs = [
    torch.tensor(aligner.get_lpz(speech[start:end]))
    for start, end in partitions["partitions"]
]
lpz = torch.cat(lpzs).numpy()
lpz = np.delete(lpz, partitions["delete_overlap_list"], axis=0)
task_overlap_n = aligner.prepare_segmentation_task(text, lpz, name, speech_len)
# Apply CTC segmentation
task_overlap_n.set(**CTCSegmentation.get_segments(task_overlap_n))
print("With 2 overlapping indices:")
print(task_overlap_n)

# make sure output has the same length
if not segments.lpz.shape[0] == task_overlap_n.lpz.shape[0]:
    print(
        f"ERROR: exp. {segments.lpz.shape[0]} != got {task_overlap_n.lpz.shape[0]} (task_overlap_n)"
    )

##### plot activations
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
        "splitted_1s": task_min.lpz[:, index],
        f"overlap_1f_{LONGEST_AUDIO_SEGMENTS}s": task_overlap.lpz[:, index],
        f"overlap_{OVERLAP}f_{LONGEST_AUDIO_SEGMENTS}s": task_overlap_n.lpz[:, index],
    }
    sns.lineplot(data=activations)
    filename = f"plot_{index}.svg"
    print("Plot: ", filename)
    plt.ylim(-25, 0)
    plt.title(f"Activations for >{name}<")
    plt.savefig(outdir / filename)
    plt.clf()
