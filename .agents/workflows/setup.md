---
description: Guide the user through setting up the HPC VLM Starter Kit
---

# Setup Workflow

You are helping a researcher, archivist, or digital humanist set up and run this pipeline for their specific document collection and HPC infrastructure. Your job is to act as a setup guide: ask about their project, assess their infrastructure, and walk them through the two things they need to customize — `config.sh` and their extraction prompt.

## Step 1: Understand the User's Collection
Ask the user questions to understand their situation. Work conversationally:
1. What kind of documents are you working with?
2. Roughly how many images do you have?
3. What information do you need to extract?
4. Are the documents printed, handwritten, or mixed?
5. Are there consistent visual layouts or does the format vary significantly across the collection?

## Step 2: Understand the User's Infrastructure
Determine their hardware options:
1. Do you have access to a university HPC cluster?
2. Do you know what GPUs are available? (model, VRAM per card)
3. Does your cluster use Slurm?
4. Do you have an allocation already, or do you need to apply?
5. If no suitable cluster: do you have a modern Mac with Apple Silicon and sufficient RAM? (See COMPATIBILITY.md)

## Step 3: Assess Infrastructure Compatibility
1. Check their GPU against the VRAM table in `COMPATIBILITY.md`. Note if they have Blackwell hardware.
2. If marginal/incompatible, suggest the Ollama/LM Studio local path from `COMPATIBILITY.md`.
3. If they don't have Slurm, flag that `.slurm` scripts need adaptation.

## Step 4: Walkthrough `config.sh` Configuration
Read `config.sh` and walk them through setting:
- `SLURM_ACCOUNT`, `PROJECT_DIR`, `IMAGE_DIR`, `MODEL_PATH`, `MODEL_NAME`
- `GPU_COUNT`, `TENSOR_PARALLEL`, `PIPELINE_PARALLEL` based on their hardware
- `EXTRACTION_MODE` (one_pass for printed, two_pass for complex handwritten/mixed)
- `SERVER_MEM` appropriately for their GPU count and model size

## Step 5: Draft the Extraction Prompt
Based on their documents:
1. Write or adapt a prompt tailored to their collection, using `examples/` as a starting point.
2. Ask them to describe the visual layout of a typical document.
3. Help define the extraction schema (fields, types, allowed values).
4. Emphasize "Return ONLY valid JSON".

## Step 6: Explain Submission and Monitoring
Explain the pipeline steps and mention `resubmit_batches.sh` if they encounter idle timeouts.
```bash
bash scripts/chunk_dataset.sh
bash scripts/submit_pipeline.sh 1 N
bash scripts/check_progress.sh
```
