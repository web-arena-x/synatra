# Code for Paper: Synatra: Turning Indirect Knowledge into Direct Demonstrations for Digital Agents at Scale

## About Synatra

- A data synthesis approach relying on indirect knowledge
- 100k next action demonstrations in the form of web trajectories
- Synatra-CodeLlama-7B, a dedicated web navigation agent

## Repository Structure

This repository is divided into two parts:

- **Data Synthesis**: contains pipeline to generate synthetic trajectories using tutorials and web page snapshots.

- **Training**: contains training code Synatra-CodeLlama-7B and all other experimented models in the paper.

- **Evaluation**: contains evaluation code on all benchmarks we tested in the paper.

## Dataset Download
- **Synatra**: Download Synatra's 100k synthesized trajectories from [huggingface](https://huggingface.co/datasets/oottyy/Synatra).

## Model Checkpoint

|  Model Name   |        LLM        |                          Checkpoint                          |
| :-----------: | :---------------: | :----------------------------------------------------------: |
|   Synatra-CodeLlama-7B   | [CodeLlama-7B](https://huggingface.co/codellama/CodeLlama-7b-hf)  | [Synatra-CodeLlama-7B](https://huggingface.co/oottyy/Synatra-Models)  |


## Data Synthesis
### Generate trajectories with WikiHow Tutorials and Web Page Snapshots
```bash
cd ./data_generation
```
Follow [instructions](https://github.com/web-arena-x/synatra/tree/main/data_generation#readme) to generate trajectories.


## Training
### Train with LLaMA-Factory
Set up LLaMA-Factory according to the [instructions](https://github.com/web-arena-x/synatra/tree/main/train/LLaMA-Factory-0.8.3#readme).

To start training:
```bash
cd ./train
python launch_training_batch.py
```


## Run Evaluation

### Serve Models With vLLM
To serve evaluated models locally with [vLLM](https://docs.vllm.ai/en/latest/):
```bash
cd ./evaluation/
sbatch vllm_serve.sh
```

### WebArena & MiniWoB++
To evaluate [WebArena](https://webarena.dev/) and [MiniWoB++](https://miniwob.farama.org/):

Use the WebArena benchmark with MiniWoB++ intergration

```bash
cd ./evaluation/webarena_miniwob
```

Follow the set-up and evaluation instruction of [webarena_miniwob](https://github.com/web-arena-x/synatra/tree/main/evaluation/webarena_miniwob#readme)


### Mind2Web Evaluation
To evaluate [Mind2Web](https://osu-nlp-group.github.io/Mind2Web/):

Run inference
```bash
cd ./evaluation/mind2web/inference

python m2w_code.py \
../data/(domain|task|website)_test.json \
MODEL_NAME \

```
Calculate metrics
```bash
python ../eval/count_m2w.py
```
