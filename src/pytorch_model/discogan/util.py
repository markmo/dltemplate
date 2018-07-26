from datetime import datetime
import json
import logging
import os


def get_time():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# noinspection SpellCheckingInspection
def prepare_dirs_and_logger(config):
    formatter = logging.Formatter('"%(asctime)s:%(levelname)s::%(message)s')
    logger = logging.getLogger()

    for handler in logger.handlers:
        logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            config.model_name = '{}_{}'.format(config.dataset, config.load_path)
    else:
        config.model_name = '{}_{}'.format(config.dataset, get_time())

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)

    config.data_path = os.path.join(config.data_dir, config.dataset)

    for path in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def save_config(config):
    param_path = os.path.join(config.model_dir, 'params.json')

    print('[*] MODEL dir: %s' % config.model_dir)
    print('[*] PARAM path: %s' % param_path)

    with open(param_path, 'w') as f:
        json.dump(config.__dict__, f, indent=4, sort_keys=True)
