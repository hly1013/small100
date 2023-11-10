"""
* this is the test code for training code before refactoring
therefore might not be executable in the current code structure *


to test:
- DataLoader
    - Sampler
    - DataSet
- for loop
    - tqdm

in train.py:
- train_dataset: ConversationEnKoDataset()
- start_idx
- from_middle_sampler: FromMiddleSequentialSampler(train_dataset, start_idx)
- train_dataloader: DataLoader(...)
"""

from train import train_dataset, FromMiddleSequentialSampler, collate_fn
from torch.utils.data import DataLoader

start_idx = 0
batch_size = 2

from_middle_sampler = FromMiddleSequentialSampler(train_dataset, start_idx)
from_middle_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    sampler=from_middle_sampler,
)
from_start_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)

# breakpoint()

# test code

### start_idx == 0
# is start same?
start_idx = 0
batch_size = 2
from_middle_sampler = FromMiddleSequentialSampler(train_dataset, start_idx)
from_middle_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    sampler=from_middle_sampler,
)
from_start_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)
import torch
first_from_start = next(iter(from_start_dataloader))
first_from_middle = next(iter(from_middle_dataloader))
for k, v in first_from_start.items():
    assert torch.equal(v, first_from_middle[k])

# is len same?
assert len(from_start_dataloader) == len(from_middle_dataloader)

# is end same?
last_from_start = None
for last_from_start in iter(from_start_dataloader):
    pass

last_from_middle = None
for last_from_middle in iter(from_middle_dataloader):
    pass

for k, v in last_from_start.items():
    assert torch.equal(v, last_from_middle[k])


### start_idx != 0
# from_start_dataloader의 start_idx * batch_size를 인덱스로 갖는 아이템이
# from_middle_dataloader의 첫번째(인덱스 0) 아이템과 동일해야 함
# 그 이후는 둘이 완전 동일해야
start_batch_idx = 5
batch_size = 13
start_idx = start_batch_idx * batch_size  # TODO: 다 이런 식으로 바꿔야 함 (sampler의 start_idx)

from_middle_sampler = FromMiddleSequentialSampler(train_dataset, start_idx)

from_middle_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    sampler=from_middle_sampler,
)
from_start_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)
# start point check
first_from_middle = next(iter(from_middle_dataloader))

from_start_iter = iter(from_start_dataloader)
for _ in range(start_batch_idx):
    next(from_start_iter)
middle_from_start = next(from_start_iter)

for k, v in first_from_middle.items():
    assert torch.equal(v, middle_from_start[k]), k

# len check
from_start_batch_len = len(from_start_dataloader) - start_batch_idx
from_middle_batch_len = len(from_middle_dataloader)
assert from_start_batch_len == from_middle_batch_len

# end point check
last_from_start = None
for last_from_start in iter(from_start_dataloader):
    pass

last_from_middle = None
for last_from_middle in iter(from_middle_dataloader):
    pass

for k, v in last_from_start.items():
    assert torch.equal(v, last_from_middle[k])

# ------------ 여기 위에까지는 잘 되는 거 확인함 --------------- #

### train.py 파일 상에서도 제대로 되는지 확인
from train import (
        batch_size,
        train_dataset,
        start_batch_idx,
        start_idx,
        start_model_name,
        from_middle_sampler,
        train_dataloader,
        RESUME_TRAINING,
        custom_enumerate,
        tqdm_kwargs,
    )

## dataloader, sampler check
# -> 위에서 한 테스트 그대로 train.py에서 적용하면 됨

## custom_enumerate, tqdm_kwargs check
# start point
assert tqdm_kwargs['initial'] == start_batch_idx == start_idx * batch_size

from_start_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)
from_start_iter = iter(from_start_dataloader)
for _ in range(start_batch_idx):
    next(from_start_iter)

a, b = next(custom_enumerate(train_dataloader))
assert a == start_batch_idx
for k, v in next(from_start_iter).items():
    assert torch.equal(v, b[k]), k

# len, end point
from tqdm import tqdm
a, b = None, None
from_middle_len = 0
for a, b in tqdm(iterable=custom_enumerate(train_dataloader), **tqdm_kwargs):
    from_middle_len += 1

last_batch = None
from_start_len = 0
for last_batch in iter(from_start_dataloader):
    from_start_len += 1

assert from_start_len == from_middle_len + start_batch_idx
assert a == tqdm_kwargs['total'] == len(from_start_dataloader)
for k, v in b.items():
    assert torch.equal(v, last_batch[k]), k
