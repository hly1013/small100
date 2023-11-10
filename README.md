# small100
fine-tune SMaLL-100 model and make inference

## Tools Used
- Hugging Face
  - model
  - tokenizer
- PyTorch
  - Dataset
  - DataLoader
  - SequentialSampler
- Python

## SMaLL-100 ?
[SMaLL-100](https://aclanthology.org/2022.emnlp-main.571/) is a distilled version of the M2M100 model.  
[M2M-100](https://arxiv.org/abs/2010.11125) is a many-to-many multilingual translation model that can translate directly between any pair of 100 languages.

## TODO
- freeze most of the model's parameters, and learn only a few parameters at fine-tuning
