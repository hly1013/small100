"""training"""

from tqdm import tqdm
import os

# for path
cur_dir_path = os.path.dirname(__file__)
if cur_dir_path == '': cur_dir_path = '.'

def main(**kwargs):
    from log import logging_train_start, logging_train_step, logging_train_end

    # from settings
    batch_size = kwargs['batch_size']
    RESUME_TRAINING = kwargs['RESUME_TRAINING']
    learning_rate = kwargs['learning_rate']

    # from items_for_train
    log_file = kwargs['log_file']
    model = kwargs['model']
    custom_enumerate = kwargs['custom_enumerate']
    tqdm_kwargs = kwargs['tqdm_kwargs']
    train_dataloader = kwargs['train_dataloader']
    optimizer = kwargs['optimizer']

    # function for training one step
    def train_step(batched_inputs):
        batched_outputs = model(**batched_inputs)

        # TODO: loss 어떻게 계산되고 있는 건지 확인
        loss = batched_outputs.loss

        loss.backward()
        optimizer.step()
        # lr_scheduler.step()
        optimizer.zero_grad()

        return loss

    # logging: start training
    logging_train_start(
        log_file,
        RESUME_TRAINING=RESUME_TRAINING,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    # training
    model.train()

    # TODO: error catch해서 예기치 못하게 학습 중단된 경우에도 모델 저장, 로깅 정상적으로 되도록
    for batch_idx, batched_inputs in tqdm(iterable=custom_enumerate(train_dataloader), **tqdm_kwargs):
        # save model: during training
        if batch_idx % 100 == 0:
            model.save_pretrained(f'{cur_dir_path}/model/{batch_idx}')

        # train one step
        loss = train_step(batched_inputs)

        # logging: during training
        if batch_idx % 10 == 0:
            logging_train_step(log_file, batch_idx=batch_idx, loss=loss)

    # save model: training end
    model.save_pretrained(f'{cur_dir_path}/model/{len(train_dataloader.dataset)}_final')

    # logging: training end
    logging_train_end(log_file)


if __name__ == "__main__":
    from setting import settings, items_for_train

    main(**settings, **items_for_train)
