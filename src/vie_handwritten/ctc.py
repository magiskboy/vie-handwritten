"""CTC loss and decoding utilities (TensorFlow)."""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf


def ctc_loss(
    y_true,
    y_pred,
    *,
    blank_index: int = 0,
    label_length=None,
    logit_length=None,
):
    """Compute mean CTC loss between label sequences and model logits."""
    y_true = tf.cast(y_true, tf.int32)
    if label_length is None:
        label_length = tf.reduce_sum(tf.ones_like(y_true), axis=1)
    if logit_length is None:
        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])

    per_example = tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred,
        label_length=tf.cast(label_length, tf.int32),
        logit_length=tf.cast(logit_length, tf.int32),
        logits_time_major=False,
        blank_index=blank_index,
    )
    # zero_infinity: when a sample's time steps can't fit its label (T < label_length,
    # or repeats needing extra blanks) CTC returns +inf; zero it so one bad sample in a
    # batch can't blow up the mean loss / gradients (mirrors torch ``zero_infinity=True``).
    per_example = tf.where(tf.math.is_finite(per_example), per_example, tf.zeros_like(per_example))
    return tf.reduce_mean(per_example)


def ctc_greedy_decode(logits: np.ndarray, *, blank_index: int = 0) -> list[list[int]]:
    """Greedy CTC decode: argmax per timestep → remove blanks & collapse repeats."""
    if logits.ndim != 3:
        raise ValueError(f"Expected logits (B, T, C), got {logits.shape}")
    batch = []
    for seq in logits:
        idxs = np.argmax(seq, axis=-1).tolist()
        collapsed: list[int] = []
        prev = None
        for i in idxs:
            if i == blank_index:
                prev = None
                continue
            if i == prev:
                continue
            collapsed.append(int(i))
            prev = i
        batch.append(collapsed)
    return batch


def ctc_beam_decode(
    logits: np.ndarray,
    *,
    blank_index: int = 0,
    beam_width: int = 10,
) -> list[list[int]]:
    """Beam-search CTC decode via TensorFlow."""
    logits_t = tf.convert_to_tensor(logits, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits_t, axis=-1)
    time_major = tf.transpose(log_probs, [1, 0, 2])
    seq_len = tf.fill([tf.shape(logits_t)[0]], tf.shape(logits_t)[1])
    decoded, _ = tf.nn.ctc_beam_search_decoder(
        time_major, seq_len, beam_width=beam_width, top_paths=1
    )
    dense = tf.sparse.to_dense(decoded[0], default_value=blank_index).numpy()
    return [[int(i) for i in row if i != blank_index] for row in dense]


def decode_predictions(
    logits: np.ndarray,
    charset: Any,
    *,
    method: str = "greedy",
    blank_index: int = 0,
    beam_width: int = 10,
) -> list[str]:
    """Decode a batch of logits into Vietnamese text strings."""
    if method == "beam":
        paths = ctc_beam_decode(logits, blank_index=blank_index, beam_width=beam_width)
    else:
        paths = ctc_greedy_decode(logits, blank_index=blank_index)
    return [str(charset.decode(path, join=True)) for path in paths]
