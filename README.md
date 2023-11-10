# small100
fine-tune SMaLL-100 model and make inference

fine-tune small100 model
and test inference; translate English sentence to Korean using small100 model

## SMaLL-100 ?
SMaLL-100 is multilingual neural machine translation model
improved from M2M-100
- model hub (hugging face): https://huggingface.co/alirezamsh/small100
- paper: https://aclanthology.org/2022.emnlp-main.571/

## Tools Used
- Hugging Face
  - model
  - tokenizer
- PyTorch
  - Dataset
  - DataLoader
  - SequentialSampler
- Python

## Directory Structure
- /data: training data
- /log: log files written during training
- /model: model checkpoints
- /utils: not included in this package automatically, but useful codes
    - clear_command.py
    - log_to_loss_plot.py
    - test_resume_training.py
- \_\_init__.py: initializer of this package. includes needed files
- \_\_main__.py: main functionality of this package. do training or inference
- tokenization_small100.py: needed for model's tokenization, provided from the small100 model developer
- training.py: fine-tuning pre-trained small100 model
- setting.py: setting needed for training
- log.py: logging during training
- inference.py: inference using trained small100 model

## How to Execute
1. start venv
    requirements installed in venv
    -->
    command `source venv/bin/activate` to start the virtual environment
    and then run files in this package
2. run main file
    command `python3 .` in terminal (at `/small100`)
    or `python3 small100` in parent directory
    to run `__main__.py`
3. type your input as instruction, and get the result!

## TODO
- freeze most of the model's parameters, and learn only a few parameters at fine-tuning
