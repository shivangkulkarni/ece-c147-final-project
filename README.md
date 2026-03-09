# EMG2QWERTY — ECE C147/C247 Final Project

Predicting QWERTY keystrokes from surface EMG signals using deep learning.

## Quick Start

```bash
# 1. Ensure baseline repo exists
#    (already included in this project as ./emg2qwerty)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Put data files in data/
#    Download subject #89335547 from the Box link in agents.md

# 4. Train required recurrent baseline (BiLSTM)
python experiments/train.py --model-type rnn --epochs 40 --batch-size 8

# 5. Evaluate saved checkpoint
python experiments/evaluate.py --checkpoint checkpoints/rnn_best.pt --split test
```

## Optional models

```bash
python experiments/train.py --model-type cnn_rnn --epochs 40 --batch-size 8

python experiments/train.py --model-type transformer --epochs 40 --batch-size 8
```

## Notes
- `experiments/configs/` and `scripts/` are optional and not required for running core experiments.
- Use `--train-fraction`, `--num-channels`, and `--downsample-factor` for quick ablations.
