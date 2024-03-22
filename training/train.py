import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
from datasets.dataset_utils import make_dataloaders
from misc.utils import UMFParams
from training.trainer import do_train
import wandb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Minkowski Net embeddings using BatchHard negative mining')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to the model-specific configuration file')
    parser.add_argument('--debug', dest='debug', action='store_true')
    # Run argument integer default 0
    parser.add_argument('--run', type=int, default=0, help='Run number')

    parser.set_defaults(debug=False)

    args = parser.parse_args()
    print('Training config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print('Debug mode: {}'.format(args.debug))

    params = UMFParams(args.config, args.model_config)
    params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    dataloaders = make_dataloaders(params, debug=args.debug)
    params.debug = args.debug

    if not args.debug:
        wandb.init(
        # Set the project where this run will be logged
        project=params.model_params.model, 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{args.run}", 
        # Track hyperparameters and run metadata
        config=params.to_dict()
        )


    do_train(dataloaders, params, debug=args.debug)
    wandb.finish()
