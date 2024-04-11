import gradio as gr

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from transformers import SegformerFeatureExtractor, TFSegformerForSemanticSegmentation

feature_extractor = SegformerFeatureExtractor.from_pretrained(
    "jonathandinu/face-parsing"
)
model = TFSegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")


def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [
        [125, 237, 123],
        [25, 97, 48],
        [59, 11, 81],
        [163, 123, 42],
        [239, 41, 136],
        [224, 4, 115],
        [114, 84, 169],
        [16, 137, 208],
        [153, 91, 30],
        [48, 90, 221],
        [91, 245, 206],
        [108, 87, 175],
        [232, 181, 231],
        [153, 70, 176],
        [32, 25, 179],
        [118, 177, 239],
        [246, 75, 15],
        [183, 17, 190],
        [79, 235, 51],
    ]


labels_list = []

with open(r"labels.txt", "r") as fp:
    for line in fp:
        labels_list.append(line[:-1])

colormap = np.asarray(ade_palette())


def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")
    return colormap[label]


def draw_plot(pred_img, seg):
    fig = plt.figure(figsize=(20, 15))

    grid_spec = gridspec.GridSpec(1, 2, width_ratios=[6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(pred_img)
    plt.axis("off")
    LABEL_NAMES = np.asarray(labels_list)
    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

    unique_labels = np.unique(seg.numpy().astype("uint8"))
    ax = plt.subplot(grid_spec[1])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation="nearest")
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0, labelsize=25)
    return fig


def sepia(input_img):
    input_img = Image.fromarray(input_img)

    inputs = feature_extractor(images=input_img, return_tensors="tf")
    outputs = model(**inputs)
    logits = outputs.logits

    logits = tf.transpose(logits, [0, 2, 3, 1])
    logits = tf.image.resize(
        logits, input_img.size[::-1]
    )  # We reverse the shape of `image` because `image.size` returns width and height.
    seg = tf.math.argmax(logits, axis=-1)[0]

    color_seg = np.zeros(
        (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
    )  # height, width, 3
    for label, color in enumerate(colormap):
        color_seg[seg.numpy() == label, :] = color

    # Show image + mask
    pred_img = np.array(input_img) * 0.5 + color_seg * 0.5
    pred_img = pred_img.astype(np.uint8)

    fig = draw_plot(pred_img, seg)
    return fig


demo = gr.Interface(
    fn=sepia,
    inputs=gr.Image(shape=(400, 600)),
    outputs=["plot"],
    examples=[
        "elon.jpg",
        "biden.jpeg",
        "bezos.jpeg",
        "zuckerberg.jpeg",
    ],
    allow_flagging="never",
)


demo.launch()
