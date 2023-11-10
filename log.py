"""logging, during training"""

import time

def logging_train_start(log_file, **kwargs):
    RESUME_TRAINING = kwargs['RESUME_TRAINING']
    batch_size = kwargs['batch_size']
    learning_rate = kwargs['learning_rate']

    current_time = time.strftime('%Y-%m-%d %H:%M %Z', time.localtime())
    log_file.write(f'training start time: {current_time}\n')
    log_file.writelines([
        f'RESUME_TRAINING: {RESUME_TRAINING}\n'
        f'batch_size: {batch_size}\n',
        f'learning_rate: {learning_rate}\n'
    ])


def logging_train_step(log_file, **kwargs):
    batch_idx = kwargs['batch_idx']
    loss = kwargs['loss']

    log_file.write(f'batch_idx: {batch_idx}\n')
    log_file.write(f'loss: {loss}\n')


def logging_train_end(log_file, **kwargs):
    current_time = time.strftime('%Y-%m-%d %H:%M %Z', time.localtime())
    log_file.write(f'training end time: {current_time}')
    log_file.close()  # TODO: log_file 안전하게 닫힐 수 있도록 with문이든 뭐 그런 거 활용
