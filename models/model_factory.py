from misc.utils import UMFParams
import dotsi
import yaml
from models.UMF.UMFnet import UMFnet


def load_config(config_file):
    with open(config_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg

def model_factory(params: UMFParams):
    in_channels = 1

    if params.model_params.model == 'UMF':
        # XMFnet baseline model
        cfg = load_config(params.model_params_path)
        cfg = dotsi.Dict(cfg)
        # add cfg to params
        params.cfg = cfg
        model = UMFnet(cfg, final_block=cfg.model.fusion.final_block)  
    else:
        raise ValueError('Unknown model: {}'.format(params.model_params.model))



    return model
