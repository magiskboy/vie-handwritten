# ResNet-18 ImageNet weights (vendored)

Upstream preset: keras-hub `resnet_18_imagenet`
(Kaggle model `keras/resnetv1`, Keras preset files `config.json` + `model.weights.h5`).

Pinned artifact: `backbone.weights.h5` — ImageNet weights remapped onto this
project's HTR ResNet-18 backbone layer names. Training loads them offline with
`Model.load_weights` (no network / keras-hub).

Upstream `model.weights.h5` SHA-256:
`2a69f8784e849bd2312fb2fd509bcdd05e76d464bfcc6b1bc9113dc37a4a496d`

## Our layer-name convention (ResNet-18 / basic block)

Stem:

| Our name   | Role                          |
| ---------- | ----------------------------- |
| `stem_conv` | 7×7 conv, stride (2, 2)      |
| `stem_bn`   | BatchNorm after stem conv    |
| `stem_relu` | (no weights)                 |
| `stem_pool` | 3×3 max-pool, stride (2, 2)  |

Each residual block is named `layer{L}_block{B}` with **1-based** `L`, `B`:

| Our suffix     | Role                                      |
| -------------- | ----------------------------------------- |
| `_conv1`       | 3×3 conv (may carry block stride)         |
| `_bn1`         | BatchNorm                                 |
| `_relu1`       | (no weights)                              |
| `_conv2`       | 3×3 conv, stride 1                        |
| `_bn2`         | BatchNorm                                 |
| `_proj_conv`   | 1×1 projection (only if downsample/chan.) |
| `_proj_bn`     | BatchNorm on projection                   |
| `_add` / `_out`| residual add + ReLU (no weights)          |

ResNet-18 layout in `build_cnn_backbone`:

- `layer1`: 2× blocks, 64 filters, stride (1, 1)
- `layer2`: 2× blocks, 128 filters; block1 stride **(2, 1)** (HTR: height÷2, width kept)
- `layer3`: 2× blocks, 256 filters; block1 stride **(2, 1)**
- `layer4`: 2× blocks, 512 filters, stride (1, 1) — no extra width downsample

Only layers that have weights are stored in `backbone.weights.h5`.

## Map from keras_hub (for re-dump / other ResNet presets)

keras_hub basic-block naming is 0-based (`stack{s}_block{b}`) with path suffixes
`_1_*` / `_2_*` and projection `_0_*`:

```
stem_conv / stem_bn          ←  conv1_conv / conv1_bn
layer{L}_block{B}_conv1      ←  stack{L-1}_block{B-1}_1_conv
layer{L}_block{B}_bn1        ←  stack{L-1}_block{B-1}_1_bn
layer{L}_block{B}_conv2      ←  stack{L-1}_block{B-1}_2_conv
layer{L}_block{B}_bn2        ←  stack{L-1}_block{B-1}_2_bn
layer{L}_block{B}_proj_conv  ←  stack{L-1}_block{B-1}_0_conv
layer{L}_block{B}_proj_bn    ←  stack{L-1}_block{B-1}_0_bn
```

Always match by **name** (not positional zip): hub emits downsample-block weights
interleaved (`_1_conv`, `_1_bn`, `_0_conv`, `_2_conv`, …).

When migrating to another ResNet (34 / 50 / …):

1. Keep or extend this naming scheme in `build_cnn_backbone`.
2. Install keras-hub temporarily, `from_preset(...)`, transfer with a map like above
   (bottleneck blocks need `_1/_2/_3` + `_0` instead of basic `_1/_2`).
3. `backbone.save_weights(...)` into a new `third_party/.../backbone.weights.h5`.
4. Point `IMAGENET_BACKBONE_WEIGHTS` at the new file and update this SOURCE.md.

Do not replace this file with an unrelated ResNet-18 checkpoint; BN moving stats
and tensor layouts must match the map above.
