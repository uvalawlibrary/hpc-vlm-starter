# HPC VLM Starter Kit

**Extract structured data from historical document images using open-source vision-language models on your university's research computing cluster.**

This starter kit provides a complete, reusable pipeline for processing scanned archival documents — index cards, court records, correspondence, registers, ledgers — through a large vision-language model (VLM) running on institutional GPU infrastructure. Developed by Loren S. Moulds and Angela Boakye Danquah, it was built from production pipelines processing millions of historical documents and historical court records. Testing and compatability evaluation was performed by Anamika Lal from Digital Research Services at University of North Carolina at Chapel Hill.

The pipeline is designed to be adapted by historians, archivists, and digital humanists who have access to a university HPC cluster but may not have built one of these systems before.

---

## What This Kit Does

1. **Serves a VLM** on GPU nodes via [vLLM](https://github.com/vllm-project/vllm), an open-source inference engine
2. **Batches your images** into manageable chunks for sequential Slurm jobs
3. **Extracts structured JSON** from each image using your custom prompt
4. **Handles failures gracefully** with skip-on-exists logic and automated retry sweeps
5. **Monitors progress** with a live terminal dashboard

You write the prompt. The pipeline handles everything else.

> **Using Claude Code?** This repository includes a `CLAUDE.md` file that instructs Claude Code to act as a setup guide for your specific project and infrastructure. Open the repo in [Claude Code](https://claude.ai/code), and it will ask about your collection, assess your cluster, walk you through `config.sh`, and help you write your extraction prompt.

> **Using Gemini or another AI assistant?** This repository includes an agent workflow at `.agents/workflows/setup.md`. Tell your assistant to **"run the setup workflow"** and it will guide you through the same onboarding process — assessing your infrastructure, configuring the pipeline, and drafting your extraction prompt.

---

## Prerequisites

- Access to an HPC cluster with Slurm and GPU nodes (A100s recommended, but any CUDA GPU with sufficient VRAM works)
- [Apptainer](https://apptainer.org/) (formerly Singularity) available on the cluster
- A vLLM container image (`.sif` file) — see [Setup](#3-get-the-container) below
- Python 3.9+ on the cluster (usually available via `module load`)
- Your scanned images in JPEG, PNG, or TIFF format

> **Not sure if your cluster is compatible?** See [COMPATIBILITY.md](COMPATIBILITY.md) for a checklist covering GPU architecture requirements, VRAM thresholds, quantization options for smaller clusters, and scheduler compatibility. Older or departmental clusters may need investigation before you begin.

### Getting HPC Access

If you do not already have an HPC allocation, expect some administrative lead time before you can run jobs. The process typically involves: creating an account with your institution's research computing group, requesting a compute allocation (usually requiring a brief research summary and computational justification), and setting up a project group if collaborators need access.

At UVA, for example, this involves [applying for an allocation](https://www.rc.virginia.edu/userinfo/hpc/access/) through the Research Computing office, establishing a Grouper group through ITS, and writing a short description of your research and computational needs. Standard allocations at UVA are free for faculty-led research projects.

Your institution's process will differ — contact your research computing office early. Most are genuinely supportive of humanities and library projects and can help you scope your resource request appropriately.

## Directory Structure

```
hpc_starterkit/
├── config.sh                          # All configuration in one place
├── slurm/
│   ├── start_server.slurm             # Launch vLLM inference server
│   ├── run_batch.slurm                # Process one batch of images
│   └── retry_failed.slurm             # Re-process failed extractions
├── scripts/
│   ├── submit_pipeline.sh             # Start server + chain all batches + retry
│   ├── resubmit_batches.sh            # Resubmit without restarting the server
│   ├── extract_one.py                 # Process a single image (also a library)
│   ├── batch_extract.py               # Threaded batch processor
│   ├── check_progress.sh              # Live status dashboard
│   └── chunk_dataset.sh               # Split images into batch directories
└── examples/
    ├── prompt_example_cards.txt      # Example: historical index cards
    ├── prompt_court_minutes.txt        # Example: Court of Session minute books
    ├── schema_example_cards.json     # JSON Schema for archival output
    └── schema_court_minutes.json       # JSON Schema for court output
```

---

## Quick Start

### 1. Clone and copy to your cluster

```bash
git clone https://github.com/YOUR_USERNAME/hpc-vlm-starter.git
scp -r hpc-vlm-starter/ you@cluster:/project/YourAllocation/code/
```

### 2. Download a model

On the cluster, download an open-weight VLM. We recommend starting with one of these:

| Model | Parameters | VRAM Required | Good For |
|-------|-----------|---------------|----------|
| **Qwen2.5-VL-7B** | 7B | ~16 GB (1 GPU) | Testing, small collections |
| **Qwen2.5-VL-72B** | 72B | ~160 GB (2-4 GPUs) | Production, complex documents |
| **Gemma 3 27B** | 27B | ~60 GB (1 GPU) | Mid-range, good accuracy/cost balance |

```bash
# Example: download Qwen2.5-VL-7B for testing
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir /project/YourAllocation/models/qwen2.5-vl-7b
```

### 3. Get the container

Build or pull a vLLM Apptainer image:

```bash
module load apptainer
apptainer pull vllm_latest.sif docker://vllm/vllm-openai:latest
mv vllm_latest.sif /project/YourAllocation/models/container/
```

### 4. Write your extraction prompt

This is the core intellectual work. Create a text file that tells the model what to extract from your documents. See `examples/` for two worked examples.

Your prompt should include:
- **What the documents are** (context helps the model)
- **What the layout looks like** (where on the page to find each field)
- **What fields to extract** (with types and allowed values)
- **Rules for ambiguous cases** (what to do when text is illegible, overlapping, etc.)
- **The output schema** (valid JSON structure)

Save it to your project directory:

```bash
nano /project/YourAllocation/code/prompts/my_prompt.txt
```

### 5. Edit config.sh

Open `config.sh` and set your values:

```bash
SLURM_ACCOUNT="YourAllocation"
MODEL_PATH="/project/YourAllocation/models/qwen2.5-vl-7b"
MODEL_NAME="my-extractor"
PROMPT_FILE="/project/YourAllocation/code/prompts/my_prompt.txt"
GPU_COUNT=1
TENSOR_PARALLEL=1
PIPELINE_PARALLEL=1
SERVER_MEM="80G"
# ... etc
```

### 6. Upload and chunk your images

```bash
# Upload images to the cluster
rsync -avz ./my_scanned_images/ you@cluster:/project/YourAllocation/images/

# Split into batch directories
ssh you@cluster "cd /project/YourAllocation && bash code/scripts/chunk_dataset.sh"
```

### 7. Submit the pipeline

```bash
ssh you@cluster "cd /project/YourAllocation && bash code/scripts/submit_pipeline.sh 1 3"
```

This starts the vLLM server, chains batch_01 through batch_03 sequentially, and queues a retry sweep at the end.

### 8. Monitor progress

```bash
ssh you@cluster "cd /project/YourAllocation && bash code/scripts/check_progress.sh"
```

Output:
```
==========================================================
           HPC VLM EXTRACTION PIPELINE STATUS
==========================================================

 Model: my-extractor
 Images: /project/YourAllocation/images

 SERVER:    RUNNING (Job 12345678, Uptime: 01:30:45)
 ACTIVE:    batch_02 (Job 12345680, Running: 00:45:12)
 EXTRACTED: 142,331 / 250,000
 PROGRESS:  [#############-----------] 56.93%
 FAILED:    23 failed extractions
```

---

## How It Works

### The Two Things You Customize

1. **`config.sh`** — paths, model, GPU count, resource limits
2. **Your prompt file** — what to extract from your specific documents

Everything else is generic infrastructure.

### Architecture

```
 SETUP (one-time)                 PIPELINE (automated)
 ================                 ====================

 ┌────────────────┐
 │ chunk_dataset  │
 │ .sh            │    ┌────────────────────┐
 │                │    │ start_server.slurm │
 │ Splits images  │    │                    │
 │ into batch_01, │    │ vLLM serves model  │
 │ batch_02, ...  │    │ on GPU node        │
 └────────────────┘    └─────────┬──────────┘
                                 │ listens on port 8000
                                 │
                       ┌─────────▼──────────┐
                       │ run_batch.slurm    │
                       │                    │◄──── batch_01, batch_02, ...
                       │ Sends images to    │      (chained via Slurm
                       │ server via HTTP,   │       dependencies)
                       │ saves JSON output  │
                       └─────────┬──────────┘
                                 │
                                 │ failures logged
                                 │
                       ┌─────────▼──────────┐
                       │ retry_failed.slurm │
                       │                    │
                       │ Aggregates failure │
                       │ logs, re-processes │
                       └────────────────────┘
```

1. **`chunk_dataset.sh`** splits your images into batch directories (symlinks, no copying)
2. **`start_server.slurm`** launches vLLM on a GPU node and writes its address to a file
3. **`run_batch.slurm`** reads the server address, then sends each image to the API via `batch_extract.py`
4. **`batch_extract.py`** uses a thread pool for concurrency, skips already-processed images, and logs failures
5. **`retry_failed.slurm`** aggregates all failure logs and reprocesses them

### One-Pass vs Two-Pass Extraction

- **One-pass** (`EXTRACTION_MODE="one_pass"`): The model sees the image and returns structured JSON directly. Simpler, faster, good for printed or clearly formatted documents.

- **Two-pass** (`EXTRACTION_MODE="two_pass"`): Pass A transcribes all visible text from the image. Pass B takes that transcription (text only, no image) and extracts structured fields. Better for complex documents with overlapping handwriting, stamps, and mixed formatting. Requires two prompt files.

### Skip-on-Exists

Every image produces a `<filename>.json` output file. Before processing, `batch_extract.py` checks if that file already exists. If it does, the image is skipped. This means:

- You can safely resubmit any batch — it will only process missing images
- If a job times out or crashes, just resubmit — no duplication
- The retry sweep catches images that failed all 3 internal retries

### Failure Handling

Each failed image is retried 3 times with exponential backoff (1s, 1.5s, 2.25s). If all retries fail, the image path and error message are logged to `failed_cards_<model>.txt`. The retry sweep aggregates all failure logs and reprocesses them in a single pass.

### Inference Settings

The two settings you are most likely to need to adjust are `TEMPERATURE` and `MAX_TOKENS` in `config.sh`.

**Temperature** controls how deterministic the model's output is. The default is `0.0`, which means the model always picks the most probable next token — maximum consistency and repeatability. This is almost always what you want for structured extraction: you are asking the model to read and transcribe, not to be creative.

The one case to raise temperature slightly is when a card or document is so ambiguous or damaged that the model at `0.0` locks into a wrong interpretation and fails every retry. Setting temperature to `0.1` or `0.2` on a retry pass introduces just enough variability to break the cycle. If you find that a significant fraction of your retry failures are the same images failing the same way, a small temperature increase is worth trying.

Avoid temperatures above `0.3` for extraction tasks — higher values introduce hallucination risk, where the model begins plausibly inventing content that is not on the page.

**`MAX_TOKENS`** caps the length of the model's response. The default of `1024` is sufficient for most index cards and short document entries. If your documents are dense — long summaries, multi-entry pages, verbose text blocks — you may need to increase this to `2048` or `4096`. Signs that `MAX_TOKENS` is too low: truncated JSON output, missing closing braces, or fields that are cut off mid-value. These show up as malformed JSON parse errors in the failure log.

**Other settings** (you will likely not need to change these):

| Setting | Default | Notes |
|---------|---------|-------|
| `top_p` | `1.0` | Nucleus sampling threshold. Leave at 1.0 when temperature is 0.0. |
| `--timeout` | `120s` | Per-request timeout. Increase for very large images or slow GPU nodes. |
| `--workers` | `32` | Concurrent requests. Reduce if you see frequent server OOM errors. |

---

## Adapting for Your Collection

### Step 1: Examine your documents

Look at 20-30 representative images. Ask yourself:
- What fields do I need? (names, dates, locations, categories, full text?)
- Where on the page does each field appear?
- What makes some images harder than others? (handwriting, damage, overlapping elements)
- What should the model do when it cannot read something?

### Step 2: Write your prompt

Start from one of the examples in `examples/`. The key sections:
- **Context**: What are these documents? What era? What institution produced them?
- **Layout guide**: Describe the spatial arrangement of information
- **Field definitions**: Name each field, its type, and allowed values
- **Ambiguity rules**: What to do with illegible text, uncertain readings
- **Output schema**: The exact JSON structure you want back

### Step 3: Test on a handful of images

Before submitting a full pipeline, test your prompt on individual images:

```bash
python scripts/extract_one.py \
    --image /path/to/sample_image.jpg \
    --endpoint http://localhost:8000/v1/chat/completions \
    --model my-extractor \
    --prompt-file prompts/my_prompt.txt
```

Review the JSON output. Iterate on your prompt until the extraction quality is acceptable. This is the step where you invest the most time — and it pays off across every image in your collection.

### Step 4: Scale up

Once your prompt works, chunk your dataset and submit the pipeline. Start with a single batch to verify at scale, then submit the rest.

---

## Cost and Resource Estimates

The figures below are approximate estimates based on our production pipelines. Your actual consumption will depend on your cluster's GPU hardware, the model you run, and your document complexity.

| Collection Size | GPU Config | Approximate Time | Approx. SUs |
|----------------|-----------|-------------------|-------------|
| 500 images | 1x A100 (7B model) | ~1 hour | ~40 SUs |
| 10,000 images | 1x A100 (27B model) | ~6 hours | ~240 SUs |
| 100,000 images | 2x A100 (72B model) | ~25 hours | ~2,000 SUs |
| 1,000,000 images | 4x A100 (72B model) | ~10 days | ~19,000 SUs |

**A note on SUs:** One SU (Service Unit) typically equals one core-hour, but the definition and pricing varies by institution. At UVA, standard research allocations are free for faculty-led projects and purchased SUs cost $0.01/SU. Your institution may define SUs differently, charge differently, or use different terminology entirely (e.g. "credits", "compute hours", "node-hours"). Check with your research computing office before estimating costs.

A 10,000-image collection on a 27B model consumes a modest fraction of a typical annual allocation at most research universities — well within what a standard free allocation can support.

Compare to commercial API pricing: 100,000 images through a cloud VLM API would cost $5,000–$15,000 at current rates. On your university cluster, the same work costs allocated compute time — which is often effectively free, and keeps your data on institutional infrastructure.

---

## Troubleshooting

**Server won't start**: Check GPU availability with `squeue` and `sinfo -p gpu`. You may need to wait for GPU nodes to free up, or reduce `GPU_COUNT`.

**Apptainer fails at container startup**: If you see errors related to home directory binding, check that `FAKEHOME_DIR` in `config.sh` is writable. The default points to `/scratch/${USER}/fakehome`; if your cluster does not have `/scratch`, change it to `${PROJECT_DIR}/fakehome`. See [COMPATIBILITY.md](COMPATIBILITY.md) for details.

**vLLM tries to contact HuggingFace at startup**: Set `HF_OFFLINE=1` in `config.sh` (the default). This prevents vLLM from checking for updates when the model is already downloaded. On clusters with restricted outbound network access, leaving this unset can cause the server to hang or fail at startup.

**OOM error when loading the model**: The model's default context window may be too large for your GPU configuration. Set `MAX_MODEL_LEN` in `config.sh` to a smaller value (e.g., `82224`) to reduce KV cache memory usage. Reduce until the model loads, at the cost of shorter maximum input length.

**`sbatch` returns a permissions error**: Remove the `#SBATCH -A ${SLURM_ACCOUNT}` line from the failing script. Some clusters do not use account-based billing and reject this directive. See [COMPATIBILITY.md](COMPATIBILITY.md).

**Partition errors on CPU-only jobs**: Remove `#SBATCH --partition=standard` from `run_batch.slurm` and `retry_failed.slurm`. Many schedulers assign a default partition automatically for non-GPU jobs.

**Extractions are slow**: Increase `WORKERS` in `config.sh`. The sweet spot is usually 16-32 concurrent requests for a 72B model on 4 GPUs.

**High failure rate**: Check `failed_cards_*.txt` for error patterns. Common causes:
- Timeout: increase `--timeout` in the batch script
- OOM on server: reduce `WORKERS` or increase server memory
- Malformed JSON output: refine your prompt (add "Return ONLY valid JSON")

**Resubmitting safely**: Use `resubmit_batches.sh` instead of `submit_pipeline.sh` to avoid restarting the server. Skip-on-exists ensures no duplication.

---

## Acknowledgments

Developed by Loren S. Moulds (University of Virginia Law Library) and Angela Boakye Danquah (Data Analytics Center, UVA), from production pipelines built for large-scale archival digitization projects at the University of Virginia. This project was supported by a [Data Analytics Center (DAC) Analytics Resource Award](https://www.rc.virginia.edu/service/dac/awards/) from UVA Research Computing. Originally published alongside the [Computational History](https://computationalhistory.substack.com) Substack.

This work builds on a growing ecosystem of AI-assisted archival projects at research universities. We are grateful for funding and support from the On the Books: AI-Assisted Collections initiative at UNC University Libraries, supported by a Mellon Foundation grant, which seeks to demonstrated the power of machine learning for unlocking historical records and increasing access to materials related to historically underrepresented communities.

## License

MIT
