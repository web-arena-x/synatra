# Code for Paper: Synatra: Turning Indirect Knowledge into Direct Demonstrations for Digital Agents at Scale

## About Synatra

- A data synthesis approach relying on indirect knowledge
- 100k next action demonstrations in the form of web trajectories
- Synatra-CodeLlama-7B, a dedicated web navigation agent

## Repository Structure

This repository is divided into two parts:

- **Train**: contains training code Synatra-CodeLlama-7B and all other experimented models in the paper.

- **Evaluation**: contains evaluation code on all benchmarks we tested in the paper.

## Dataset Download
- **Synatra**: Download Synatra's 100k synthesized trajectories from [huggingface](https://huggingface.co/datasets/oottyy/Synatra).

## Model Checkpoint

|  Model Name   |        LLM        |                          Checkpoint                          |
| :-----------: | :---------------: | :----------------------------------------------------------: |
|   Synatra-CodeLlama-7B   | [CodeLlama-7B](https://huggingface.co/codellama/CodeLlama-7b-hf)  | [Synatra-CodeLlama-7B](https://huggingface.co/oottyy/Synatra-Models)  |
