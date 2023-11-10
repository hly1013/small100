"""main functionality of this package"""

# do training? or inference?
option = input("type 't' for training\n" +
               "type 'i' for inference\n")

# do training
if option == 't':
    import training as train
    from setting import settings, setting_main

    # * user input * #
    settings = dict()  # TODO: get settings through json or yaml or ...

    settings['batch_size'] = int(input('batch_size (int): '))

    settings['RESUME_TRAINING'] = input('RESUME_TRAINING (bool): ')
    if settings['RESUME_TRAINING'] == 'True':
        settings['RESUME_TRAINING'] = True
    elif settings['RESUME_TRAINING'] == 'False':
        settings['RESUME_TRAINING'] = False
    else:
        raise Exception

    settings['start_batch_idx'] = int(input('start_batch_idx (int): '))
    settings['start_model_name'] = input('start_model_name (str): ')
    settings['learning_rate'] = float(input('learning_rate (float): '))
    settings['log_file_name'] = input('log_file_name (str): ')

    from torch.optim import AdamW
    settings['optimizer_function'] = AdamW  # TODO: optimizer function -> how to ?

    items_for_train = setting_main(**settings)

    # train
    train.main(**settings, **items_for_train)

# do inference
elif option == 'i':
    import inference

    # * user input * #
    FINE_TUNED = input("FINE_TUNED: type 't' for True, 'f' for False\n")
    if FINE_TUNED == 't':
        FINE_TUNED = True
    elif FINE_TUNED == 'f':
        FINE_TUNED = False
    else:
        raise Exception

    if FINE_TUNED:
        model_name = input("model_name: type model name to use\n")
    else:
        model_name = None

    # inference
    settings = {'FINE_TUNED': FINE_TUNED, 'model_name': model_name}
    inference.main(**settings)
