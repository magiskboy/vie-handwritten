"""CTC loss and decoding utilities (TensorFlow)."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import tensorflow as tf


def ctc_requires_cpu() -> bool:
    """Metal (Apple Silicon) lacks OpKernels used by dense CTC (e.g. ``IsFinite``).

    Force CTC onto CPU on Darwin; keep GPU path on CUDA/Linux.
    See Apple Metal PluggableDevice gaps and TF ``ctc_loss_dense`` internals.
    """
    return sys.platform == "darwin"


def ctc_loss(
    y_true,
    y_pred,
    *,
    blank_index: int = 0,
    label_length=None,
    logit_length=None,
):
    """Compute mean CTC loss between label sequences and model logits.

    On macOS/Metal, the loss runs under ``/CPU:0`` because ``tf.nn.ctc_loss``
    dense path needs ``IsFinite`` / ``ReduceLogSumExp`` kernels that Metal
    does not register. Forward CNN/Transformer can still run on Metal GPU.
    """
    y_true = tf.cast(y_true, tf.int32)
    if label_length is None:
        label_length = tf.reduce_sum(tf.ones_like(y_true), axis=1)
    if logit_length is None:
        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])

    label_length = tf.cast(label_length, tf.int32)
    logit_length = tf.cast(logit_length, tf.int32)

    def _compute(labels, logits, lab_len, log_len):
        per_example = tf.nn.ctc_loss(
            labels=labels,
            logits=logits,
            label_length=lab_len,
            logit_length=log_len,
            logits_time_major=False,
            blank_index=blank_index,
        )
        return tf.reduce_mean(per_example)

    if ctc_requires_cpu():
        with tf.device("/CPU:0"):
            return _compute(
                tf.identity(y_true),
                tf.identity(y_pred),
                tf.identity(label_length),
                tf.identity(logit_length),
            )
    return _compute(y_true, y_pred, label_length, logit_length)


def ctc_greedy_decode(
    logits: np.ndarray,
    *,
    blank_index: int = 0,
    input_lengths=None,
) -> list[list[int]]:
    """Greedy CTC decode: argmax per timestep → remove blanks & collapses.

    ``input_lengths`` (per-sample valid timesteps) truncates each sequence so
    padded regions in a batch do not emit spurious characters.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits (B, T, C), got {logits.shape}")
    if input_lengths is not None:
        input_lengths = np.asarray(input_lengths).reshape(-1)
    batch = []
    for i, seq in enumerate(logits):
        if input_lengths is not None:
            valid = max(1, min(int(input_lengths[i]), seq.shape[0]))
            seq = seq[:valid]
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
    input_lengths=None,
) -> list[list[int]]:
    """Beam-search CTC decode via TensorFlow (CPU-safe on Metal).

    ``input_lengths`` (per-sample valid timesteps) bounds the beam search so
    padded regions in a batch are ignored.
    """
    logits_t = tf.convert_to_tensor(logits, dtype=tf.float32)
    max_t = tf.shape(logits_t)[1]
    if input_lengths is None:
        seq_len_t = tf.fill([tf.shape(logits_t)[0]], max_t)
    else:
        seq_len_t = tf.cast(
            tf.reshape(tf.convert_to_tensor(input_lengths), [-1]), tf.int32
        )
        seq_len_t = tf.clip_by_value(seq_len_t, 1, max_t)

    def _decode():
        log_probs = tf.nn.log_softmax(logits_t, axis=-1)
        time_major = tf.transpose(log_probs, [1, 0, 2])
        seq_len = seq_len_t
        decoded, _ = tf.nn.ctc_beam_search_decoder(
            time_major,
            seq_len,
            beam_width=beam_width,
            top_paths=1,
        )
        return tf.sparse.to_dense(decoded[0], default_value=blank_index)

    if ctc_requires_cpu():
        with tf.device("/CPU:0"):
            dense = _decode()
    else:
        dense = _decode()
    dense_np = dense.numpy()
    results: list[list[int]] = []
    for row in dense_np:
        results.append([int(i) for i in row if i != blank_index])
    return results


def decode_predictions(
    logits: np.ndarray,
    charset: Any,
    *,
    method: str = "greedy",
    blank_index: int = 0,
    beam_width: int = 10,
    input_lengths=None,
) -> list[str]:
    """Decode a batch of logits into Vietnamese text strings.

    ``input_lengths`` (per-sample valid timesteps) prevents padded regions in a
    batch from producing spurious characters.
    """
    if method == "beam":
        paths = ctc_beam_decode(
            logits,
            blank_index=blank_index,
            beam_width=beam_width,
            input_lengths=input_lengths,
        )
    else:
        paths = ctc_greedy_decode(
            logits, blank_index=blank_index, input_lengths=input_lengths
        )
    texts: list[str] = []
    for path in paths:
        texts.append(str(charset.decode(path, join=True)))
    return texts
