from pytorch_model.discogan.config import get_config
from pytorch_model.discogan.data_loader import get_loader
from pytorch_model.discogan.trainer import Trainer
from pytorch_model.discogan.util import prepare_dirs_and_logger, save_config
import torch


def run(conf):
    prepare_dirs_and_logger(conf)
    torch.manual_seed(conf.random_seed)
    if conf.num_gpu > 0:
        torch.cuda.manual_seed(conf.random_seed)

    if conf.is_train:
        data_path = conf.data_path
        batch_size = conf.batch_size
    else:
        if conf.test_data_path is None:
            data_path = conf.data_path
        else:
            data_path = conf.test_data_path

        batch_size = conf.sample_per_image

    a_data_loader, b_data_loader = get_loader(data_path, batch_size, conf.input_scale_size,
                                              conf.num_worker, conf.skip_pix2pix_processing)

    trainer_ = Trainer(conf, a_data_loader, b_data_loader)

    if conf.is_train:
        save_config(conf)
        trainer_.train()
    else:
        if not conf.load_path:
            raise Exception('[!] You should specify `load_path` to load a pretrained model')

        trainer_.test()


if __name__ == '__main__':
    cfg, _ = get_config()
    run(cfg)
