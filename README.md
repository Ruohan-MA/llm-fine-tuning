# LLM Classification Fine-Tuning

Fine-tuning Gemma-2-9B-IT for the [LMSYS Chatbot Arena Human Preference Predictions](https://www.kaggle.com/competitions/lmsys-chatbot-arena) Kaggle competition. Given a prompt and two model responses, predict which response human evaluators preferred (model A wins, model B wins, or tie).

## Notebooks

- **`llm-classification-finetuning.ipynb`** — Fine-tunes Gemma-2-9B-IT with LoRA (PEFT) for 3-class sequence classification. Runs on RunPod with an A100/H100 GPU.
- **`generate-submission-from-adapter.ipynb`** — Loads the trained LoRA adapter and runs inference on the test set to produce `submission.csv`. Runs on Kaggle (T4 x2).

## Setup

### Training (RunPod)

Connect via SSH:
```bash
ssh runpod
```

Run the training notebook in a tmux session:
```bash
tmux new -s train
jupyter nbconvert --to notebook --execute --inplace llm-classification-finetuning.ipynb
```

### Inference (Kaggle)

1. Upload `adapter.zip` (trained LoRA weights) and `test.csv` to the Kaggle notebook's working directory.
2. Run `generate-submission-from-adapter.ipynb`.

The notebook auto-detects whether it's running on Kaggle (uses local Gemma-2 model at `/kaggle/input/models/google/gemma-2/pytorch/gemma-2-9b-it/1`) or elsewhere (downloads from HuggingFace Hub).

## Model

- **Base model**: `google/gemma-2-9b-it`
- **Fine-tuning**: LoRA via PEFT
- **Quantization**: 4-bit NF4 (bitsandbytes) for inference
- **Labels**: `winner_model_a`, `winner_model_b`, `winner_tie`