# Cluster Compatibility Guide

This guide helps you assess whether your institution's HPC cluster can run this pipeline before you invest time in setup. It covers GPU requirements, VRAM thresholds, quantization options for smaller clusters, and job scheduler compatibility.

If you are unsure about any of these, your research computing help desk can confirm GPU generation, VRAM per card, and scheduler type in a single email.

---

## GPU Architecture and VRAM

vLLM requires NVIDIA GPUs with CUDA compute capability **sm_70 or higher** (Volta architecture, 2017 or newer). Older GPUs — including Kepler (K20, K80), Maxwell, and Pascal-generation cards — will not run vLLM regardless of VRAM.

To check what GPUs your cluster has:
```bash
sinfo -o "%n %G" | grep gpu
```

### VRAM by Model Size

| VRAM Available | Viable Models |
|----------------|---------------|
| < 16 GB | Not recommended |
| 16 GB (e.g. A2, T4) | 7B models only, no headroom |
| 40 GB (e.g. A100 40GB) | Up to 13B comfortably; 27B with quantization |
| 80 GB (e.g. A100 80GB) | Up to 27B; 72B with 2–4 GPUs |
| 80 GB × 4 | 72B models comfortably |

### Quantization for Smaller Clusters

If your cluster only has smaller or older GPUs, quantization can reduce VRAM requirements significantly — often cutting memory use roughly in half at modest accuracy cost. This is worth exploring for 7B and 13B models on 8–16 GB cards.

Add the `--quantization` flag to the `vllm serve` command in `slurm/start_server.slurm`:

```bash
vllm serve /path/to/model \
    --served-model-name my-extractor \
    --quantization awq \          # or gptq
    --dtype auto \
    --host 0.0.0.0 \
    --port 8000
```

AWQ and GPTQ quantized versions of popular models are available on HuggingFace (search for model names with `-AWQ` or `-GPTQ` suffixes).

---

## Job Scheduler

The Slurm scripts in this kit (`slurm/*.slurm`) use `sbatch`, `squeue`, and `#SBATCH` directives and are specific to **Slurm**. If your cluster uses a different scheduler, the scripts will need to be adapted:

| Scheduler | Used At | Adaptation Needed |
|-----------|---------|-------------------|
| **Slurm** | Most major research universities | None — works as-is |
| **PBS/Torque** | Some older clusters | Replace `#SBATCH` with `#PBS` directives |
| **LSF** | Some HPC centers | Replace with `#BSUB` directives |
| **SGE/UGE** | Some departmental clusters | Replace with `#$ -` directives |
| **Scyld/proprietary** | Some smaller institutional clusters | Significant adaptation required |

The Python extraction scripts — `scripts/extract_one.py` and `scripts/batch_extract.py` — are scheduler-agnostic and will work on any system. Only the job submission wrappers need to change.

---

## Examples: Smaller Institutional Clusters

Not all university clusters are large research computing facilities. Departmental and teaching clusters at smaller institutions may have significant constraints:

**JMU (NVIDIA A2, 16 GB VRAM, Slurm)**
The scripts will run, but only a 7B model fits. Expect slower throughput and less headroom for concurrent requests. Reduce `WORKERS` in `config.sh` to 4–8. Quantization may help.

**WMU Thor (NVIDIA K20, 5 GB VRAM, Scyld scheduler)**
This cluster is not compatible with vLLM. The K20 is a Kepler-generation GPU (sm_35), below the sm_70 requirement. The scheduler is also not Slurm. This pipeline cannot run on this infrastructure without a significant hardware upgrade.

If you are at a smaller institution with limited GPU resources, consider:
- Applying for an XSEDE/ACCESS allocation — national HPC resources available to researchers at any US institution
- Collaborating with a larger research university that has more capable infrastructure
- Contacting your research computing office about GPU upgrade plans or cloud burst options

---

## When a Laptop Beats the Cluster

If your institutional cluster has older GPUs, insufficient VRAM, or a non-Slurm scheduler — or if your collection is relatively small — a modern laptop may genuinely be the better choice.

Apple Silicon Macs (M4, M4 Max, M5, and later) use a unified memory architecture where CPU and GPU share the same memory pool. A MacBook Pro with 64 GB of unified memory can run a 27B model entirely in RAM; 128 GB opens up larger models. There is no separate VRAM constraint. Critically, both [Ollama](https://ollama.com) and [LM Studio](https://lmstudio.ai) — free, easy-to-install local model runners — expose an **OpenAI-compatible API endpoint**, which means `batch_extract.py` works without a single script change. Just point it at `http://localhost:11434/v1/chat/completions` (Ollama) or `http://localhost:1234/v1/chat/completions` (LM Studio) instead of your cluster.

### When local is the right call

- **Collection under ~5,000 images**: Queue wait time on a small cluster likely exceeds local processing time
- **Incompatible cluster infrastructure**: Older GPUs, wrong scheduler, or no GPU nodes at all
- **Privacy-sensitive materials**: Data never leaves your machine — no institutional network, no shared storage
- **Iterating on your prompt**: Much faster to test and refine locally before committing to a cluster run
- **No HPC allocation yet**: Start processing immediately while paperwork clears

### Rough throughput on Apple Silicon

| Hardware | Model | Approx. Throughput |
|----------|-------|--------------------|
| M4 Pro, 48 GB | 7B | ~8–12 images/min |
| M4 Max, 64 GB | 27B | ~2–4 images/min |
| M5, 128 GB | 27B | ~4–6 images/min |

At 3 images/minute on a 27B model, 5,000 images completes overnight (~28 hours). For 50,000 images, the cluster becomes the better tool.

### Getting started with Ollama

```bash
# Install Ollama (Mac/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a vision-capable model
ollama pull qwen2.5vl:7b      # 7B — fast, fits 16 GB+
ollama pull qwen2.5vl:32b     # 32B — better quality, needs 64 GB+

# Ollama serves on port 11434 by default
# Point batch_extract.py at it:
python scripts/batch_extract.py \
    --input-dir ./my_images/ \
    --output-dir ./outputs/ \
    --endpoint http://localhost:11434/v1/chat/completions \
    --model qwen2.5vl:7b \
    --prompt-file prompts/my_prompt.txt \
    --workers 2
```

Keep `--workers` low (2–4) on local runs — you are the only user, and the model is already saturating your hardware.

---

## Quick Checklist

Before starting setup, verify:

- [ ] Cluster uses Slurm (or you are prepared to adapt the scripts)
- [ ] GPUs are NVIDIA, Volta (2017) or newer (sm_70+)
- [ ] At least 16 GB VRAM available per GPU node
- [ ] Apptainer or Singularity available (`apptainer --version`)
- [ ] Python 3.9+ available via module system (`module avail python`)
- [ ] Sufficient project storage for model weights (14–150 GB depending on model) and output JSON files
