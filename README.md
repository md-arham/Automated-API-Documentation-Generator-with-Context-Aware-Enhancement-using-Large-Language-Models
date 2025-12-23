# Enhancing Translation-Based API Documentation with Large Language Models

## Overview

This repository contains the implementation and evaluation code for a research project that enhances the translation-based API documentation framework (**gDoc**) using **Large Language Models (LLMs)**. We replace traditional Seq2Seq models with **Llama-3-8B**, fine-tuned using **Quantized Low-Rank Adaptation (QLoRA)**, to generate more semantic and context-aware API documentation from **OpenAPI specifications**.

The project evaluates the effectiveness of LLMs in generating high-quality API documentation under **zero-shot**, **few-shot**, and **fine-tuned** settings, and compares them against a T5-based Seq2Seq baseline.

---

## Key Contributions

- Replication of the **T5-Base Seq2Seq baseline** used in the original gDoc framework  
- Integration of **Llama-3-8B-Instruct** with **4-bit QLoRA fine-tuning**  
- Evaluation across **zero-shot**, **few-shot**, and **fine-tuned** LLM settings  
- Comprehensive comparison using **ROUGE**, **BLEU**, and **BERTScore**  
- Demonstration of the **semantic superiority** of LLMs over lexical translation models  

---

## System Architecture

OpenAPI Specification
↓
Metadata Extraction
↓
Input Linearization
↓
┌──────────────────┐ ┌────────────────────────┐
│ T5-Base Seq2Seq │ │ Llama-3-8B + QLoRA │
└──────────────────┘ └────────────────────────┘
↓ ↓
Generated API Documentation


---

## Dataset

- **Source:** `dlt-hub/openapi-specs`
- **Examples:** 7,927 API operation–description pairs  
- **Split:** 80% Train / 10% Validation / 10% Test  
- **Scope:** Endpoint descriptions only (schemas excluded)

📁 See: `data/README.md` for preprocessing and dataset details.

---

## Models

### Baseline Model — T5-Base (Seq2Seq)
- Full fine-tuning (220M parameters)
- Optimized for lexical overlap
- Serves as the gDoc reference baseline

### Enhanced Model — Llama-3-8B + QLoRA
- 4-bit NF4 quantization
- LoRA adapters (rank = 16, α = 32)
- Instruction-tuned as a *technical documentation writer*
- Enables large-model fine-tuning on limited GPU resources

---

## Results Summary

| Model | Setting | ROUGE-1 | BLEU | BERTScore |
|------|--------|--------|------|-----------|
| T5-Base | Fine-tuned | 36.54 | 2.77 | 80.97 |
| Llama-3-8B | Zero-shot | 33.17 | 4.43 | 87.43 |
| Llama-3-8B | Few-shot | 35.44 | 6.35 | 88.94 |
| **Llama-3-8B + QLoRA** | **Fine-tuned** | **34.44** | **22.34** | **89.54** |

---

## Installation

git clone https://github.com/md-arham/Automated-API-Documentation-Generator-with-Context-Aware-Enhancement-using-Large-Language-Models.git
cd gdoc-llm-enhancement 
pip install -r requirements.txt

##Training
Train T5 Baseline
python models/t5_baseline/train_t5.py

Fine-Tune Llama-3 with QLoRA
python models/llama3_qlora/train_qlora.py


##Hardware Used

Dual NVIDIA T4 GPUs (32 GB VRAM total)

##Kaggle environment

4-bit NF4 quantization with bf16 compute

Model Weights

Due to GitHub file size limitations, trained model weights (~8 GB) are not stored in this repository.

Instructions to reproduce training or download weights (if required) are provided in the documentation and configuration files.

###Lessons Learned

Semantic metrics are essential for evaluating LLM-generated documentation

Data preprocessing quality significantly impacts downstream performance

QLoRA enables billion-parameter model fine-tuning on consumer-grade GPUs
