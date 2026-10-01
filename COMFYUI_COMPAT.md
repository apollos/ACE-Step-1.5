# Local ComfyUI / Transformers 5 compatibility

Validated on 2026-10-01 on the WorkSpace workstation. This is a local XL-SFT
compatibility patch based on upstream commit `ca1e85f`, not a general claim of
Transformers 5 support across all ACE-Step models or training configurations.

## Source and runtime

- Personal branch: `apollos/ACE-Step-1.5`, `compat/comfyui-transformers5`.
- Source: `~/Workspace/ACE-Step-1.5`.
- Interpreter: `~/.venv/comfyui/bin/python`, Python 3.12.3.
- GPU: AMD Radeon RX 7900 XTX, 24 GiB; ROCm 7.2.
- PyTorch: `2.14.0+rocm7.2`; torchaudio: `2.11.0+rocm7.2`.
- Transformers: `5.17.0`; diffusers: `0.37.1`.
- vector-quantize-pytorch: `1.31.6`; rotary-embedding-torch: `0.8.9`.
- NumPy: `2.5.3`; soundfile: `0.14.0`.

The final workflow runs inference directly inside the existing ComfyUI process
and virtual environment. The earlier `acestep-test` environment was used for
experiments and is not a runtime dependency of the ComfyUI nodes.
Keep the validated ROCm and Transformers packages when reproducing this setup;
the upstream requirements are not a lock file for this particular environment.

## Why the model code was changed

Only `acestep/models/xl_sft/modeling_acestep_v15_xl_base.py` is patched.

1. **ResidualFSQ construction:** this module evaluates tensor values and creates
   non-persistent buffers. Construct this small module under
   `torch.device("cpu")` even when Transformers uses meta initialization.
   Normal model placement moves it to the inference device afterwards.
2. **Buffer restoration after loading:** in the observed Transformers 5 loading
   path, non-persistent buffers were materialized without their required values.
   They are not restored from checkpoint tensors. Delegate Qwen3RotaryEmbedding
   initialization to the parent initializer to rebuild its frequency buffers.
   For ResidualFSQ, construct a CPU reference module and copy its buffers to the
   destination device and dtype. Learned projection parameters are preserved;
   the reference construction preserves CPU RNG state with `fork_rng(devices=[])`.

The constructor fix alone allowed the weights to load but was not sufficient:
a 55-second generation produced all-NaN latents. Both changes are required for
the validated workflow. The node also checks the restored buffer values after
loading, before attempting generation.

ACE-Step's model-code synchronization copies the patched Python code into the
local model directory when needed. It does not change the model weights.

## Local model components and inference settings

The model root is `~/local-model-workbench/models/ACE-Step/Ace-Step1.5`:

- `acestep-v15-xl-sft`
- `vae`
- `Qwen3-Embedding-0.6B`

The tested workflow uses BF16, SDPA, Euler ODE, 75 steps, CFG 7.0, cover strength
0.5 and shift 1.0. The DiT stays on GPU; the VAE and text encoder use the tested
CPU offload path. Quantization, torch.compile, LM/CoT and adapters are disabled.
The node requires the local files and does not download models during inference.

## Validation and workflow

After applying both fixes, the user confirmed successful 55-second generation
in the ComfyUI virtual environment, successful Cover node execution, successful
MelBand separation/mixing, and successful execution of the complete chain in
ComfyUI: input audio, Cover, separation, mixing, and WAV/parameter saving.
The complete-chain input was 55.709 seconds. The node's accepted 10-480 second
input range is not a claim that every duration has been tested end to end.

The style prompt is editable. A smoky male folk-rock voice is an example preset,
not a restriction of this compatibility patch. The current Cover node sets
`instrumental=False` and `vocal_language="zh"`. Full-song use requires the full
audio and its corresponding lyrics, with runtime and memory verified separately.

Nodes, tests, workflow examples and further setup details are maintained in
[local-comfyui-nodes at e81861c](https://github.com/apollos/local-comfyui-nodes/tree/e81861c/ComfyUI-MusicRemake).
This ACE-Step repository keeps the model compatibility change and its rationale.

The workstation's ComfyUI launcher retains `--disable-async-offload`,
`--disable-pinned-memory` and `--disable-dynamic-vram`, including the previously
validated realtime-audio stability configuration.

Weights, source songs, generated audio and runtime caches are stored outside
Git. Development changes are tested on the local workstation, confirmed by the
user, then committed and pushed from that same local repository.
