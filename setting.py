"""settings, for training"""

# for path
import os
cur_dir_path = os.path.dirname(__file__)

def setting_main(**kwargs):
    from transformers import M2M100ForConditionalGeneration
    from tokenization_small100 import SMALL100Tokenizer

    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    import pandas as pd
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import SequentialSampler
    import math

    class ConversationEnKoDataset(Dataset):
        def __init__(self):
            self.data = pd \
                .read_excel(f'{cur_dir_path}/data/2_대화체.xlsx')[['원문', '번역문']] \
                .rename(columns={'원문': 'target', '번역문': 'source'})

            self.tokenizer = SMALL100Tokenizer.from_pretrained(
                pretrained_model_name_or_path="alirezamsh/small100",
                tgt_lang="ko",
                truncation=True)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            d = self.data.iloc[idx]
            model_inputs = self.tokenizer(d.source, text_target=d.target, return_tensors='pt')
            return model_inputs

    # TODO: collate_fn 최적화 -> collate_fn 테스트 코드 작성
    def collate_fn(batch):
        keys = list(batch[0].keys())
        result = {key: [] for key in keys}

        for one in batch:
            for key in keys:
                result[key].append(one[key][0])

        for k, v in result.items():
            result[k] = pad_sequence(v, batch_first=True)

        return result

    class FromMiddleSequentialSampler(SequentialSampler):
        """idx of data_source (include): start_idx ~ len(data_source) - 1"""

        def __init__(self, data_source, start_idx):
            super().__init__(data_source)  # self.data_source = data_source
            self.start_idx = start_idx

        def __iter__(self):
            # [ start_idx, len(data_source) )
            return iter(range(self.start_idx, len(self.data_source)))

        def __len__(self):
            length = len(range(self.start_idx, len(self.data_source)))
            assert len(self.data_source) - self.start_idx == length
            return length


    batch_size = kwargs['batch_size']
    RESUME_TRAINING = kwargs['RESUME_TRAINING']
    start_batch_idx = kwargs['start_batch_idx']
    start_model_name = kwargs['start_model_name']
    learning_rate = kwargs['learning_rate']
    log_file_name = kwargs['log_file_name']
    optimizer_function = kwargs['optimizer_function']

    if RESUME_TRAINING:
        start_idx = start_batch_idx * batch_size
    else:
        start_batch_idx = start_idx = 0

    # logging
    log_file = open(f'{cur_dir_path}/log/{log_file_name}.txt', 'a')

    # TODO: hyperparameter tuning (especially learning rate)

    train_dataset = ConversationEnKoDataset()

    if RESUME_TRAINING:
        from_middle_sampler = FromMiddleSequentialSampler(train_dataset, start_idx)
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            sampler=from_middle_sampler,
        )

        model = M2M100ForConditionalGeneration.from_pretrained(f"{cur_dir_path}/model/{start_model_name}")

    else:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")

    '''
    # scheduler needs `resuming training from checkpoint` also
    from transformers import get_scheduler
    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    '''

    optimizer = optimizer_function(
        params=model.parameters(),
        lr=learning_rate
    )

    if RESUME_TRAINING:
        def custom_enumerate(dataloader):
            idx_iterable = range(start_batch_idx, len(dataloader))
            return zip(idx_iterable, dataloader)
    else:
        custom_enumerate = enumerate

    tqdm_kwargs = {
        'total': math.ceil(len(train_dataloader.dataset) / batch_size),  # len(train_dataloader),
        'unit': 'batch',
        'desc': 'training',
        'initial': start_batch_idx if RESUME_TRAINING else 0,
    }

    items_for_train = {
        'log_file': log_file,
        'model': model,
        'custom_enumerate': custom_enumerate,
        'tqdm_kwargs': tqdm_kwargs,
        'train_dataloader': train_dataloader,
        'optimizer': optimizer,
    }
    return items_for_train


from torch.optim import AdamW

settings = {
    'batch_size': 64,
    'RESUME_TRAINING': True,
    'start_batch_idx': 100,
    'start_model_name': '100',  # str(start_batch_idx)
    'learning_rate': 1e-4,  # default: 1e-3
    'log_file_name': 'log',
    'optimizer_function': AdamW,
}
items_for_train = setting_main(**settings)
