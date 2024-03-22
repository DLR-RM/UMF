import os
import time
import torch
import yaml


class ModelParams:
    def __init__(self, model_params_path):
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(model_params_path)
        with open(model_params_path, 'r') as f:
            params = yaml.safe_load(f)

        self.model_params_path = model_params_path
        self.params = params.get('model')
        self.model = self.params.get("name")

    def print(self):
        print('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            print('{}: {}'.format(e, param_dict[e]))

        print('')

    def to_dict(self):
        param_dict = {}
        for e in param_dict:
            param_dict[e] = getattr(self, e)

def get_datetime():
    return time.strftime("%Y%m%d_%H%M")


class UMFParams:
    """
    Params for training MinkLoc models on Oxford dataset
    """
    def __init__(self, params_path, model_params_path=None):
        """
        Configuration files
        :param path: General configuration file
        :param model_params: Model-specific configuration
        """

        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path

        with open(self.params_path, 'r') as f:
            config = yaml.safe_load(f)

        params = config['DEFAULT']

        self.dataset_path =  params.get('dataset_path')
        self.dataset =  params.get('dataset', 'robotcar')

        self.num_points = int(params.get('num_points', 4096))
        self.dataset_folder = params.get('dataset_folder')
        self.use_cloud = bool(params.get('use_cloud', True))
        self.debug = bool(params.get('debug', False))
        if self.debug:
            torch.autograd.set_detect_anomaly(True)
            print('Debug mode: ON')

        # Train with RGB
        # Evaluate on Oxford only (no images for InHouse datasets)
        self.use_rgb = True
        self.image_path = params.get('image_path')

        if self.dataset.lower() == 'robotcar':
            if 'lidar2image_ndx_path' not in params:
                self.lidar2image_ndx_path = os.path.join(self.image_path, 'lidar2image_ndx.pickle')
            else:
                self.lidar2image_ndx_path = params.get('lidar2image_ndx_path')


            self.eval_database_files = ['oxford_evaluation_database.pickle']
            self.eval_query_files = ['oxford_evaluation_query.pickle']
            
        elif self.dataset.lower() == 'etna':
            self.eval_database_files = ["etna_evaluation_database.pickle"]
            self.eval_query_files = ["etna_evaluation_query.pickle"]
            
        assert len(self.eval_database_files) == len(self.eval_query_files)

        params = config['TRAIN']
        self.num_workers = int(params.get('num_workers', 0))
        self.train_step = params.get('train_step', 'single_step')
        self.batch_size = int(params.get('batch_size', 128))
        # Validation batch size is fixed and does not grow
        self.val_batch_size = int(params.get('val_batch_size', 64))

        # Set batch_expansion_th to turn on dynamic batch sizing
        # When number of non-zero triplets falls below batch_expansion_th, expand batch size
        self.batch_expansion_th = float(params.get('batch_expansion_th', None))
        if self.batch_expansion_th is not None:
            assert 0. < self.batch_expansion_th < 1., 'batch_expansion_th must be between 0 and 1'
            self.batch_size_limit = int(params.get('batch_size_limit', 256))
            # Batch size expansion rate
            self.batch_expansion_rate = float(params.get('batch_expansion_rate', 1.5))
            assert self.batch_expansion_rate > 1., 'batch_expansion_rate must be greater than 1'
        else:
            self.batch_size_limit = self.batch_size
            self.batch_expansion_rate = None

        self.lr = float(params.get('lr', 1e-3))
        # lr for image feature extraction
        self.image_lr = float(params.get('image_lr', 1e-4))

        self.load_weights = None
        if "load_weights" in params:
            self.load_weights = params.get('load_weights')

        self.optimizer = params['optimizer']

        self.scheduler = params.get('scheduler', 'MultiStepLR')
        if self.scheduler is not None:
            if self.scheduler == 'CosineAnnealingLR':
                self.min_lr = float(params.get('min_lr',  1e-5))

            elif self.scheduler == 'WarmupCosineSchedule':
                self.warmup_steps = int(params.get('warmup_steps', 10))

            elif self.scheduler == 'LinearWarmupCosineAnnealingLR':
                self.warmup_epochs = int(params.get('warmup_epochs', 10))
            elif self.scheduler == 'OneCycleLR':
                pass
            elif self.scheduler == 'MultiStepLR':
                scheduler_milestones = params.get('scheduler_milestones')
                if not isinstance(scheduler_milestones, list):
                    self.scheduler_milestones = scheduler_milestones
                else:
                    self.scheduler_milestones = [int(e) for e in scheduler_milestones]
            elif self.scheduler == 'ReduceLROnPlateau':
                self.patience = int(params.get('patience', 2))
                self.factor = float(params.get('factor', 0.9))
            elif self.scheduler == 'ExpotentialLR':
                self.gamma = float(params.get('gamma', 0.95))
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.scheduler))

        self.epochs = int(params.get('epochs', 20))
        self.weight_decay = float(params.get('weight_decay', None))
        self.normalize_embeddings = bool(params.get('normalize_embeddings', True))    # Normalize embeddings during training and evaluation
        self.loss = params.get('loss')

        weights = params.get('weights', [0.3, 0.3, 0.3])
        self.weights = [float(e) for e in weights]

        if 'Contrastive' in self.loss:
            self.pos_margin = float(params.get('pos_margin', 0.2))
            self.neg_margin = float(params.get('neg_margin', 0.65))
        elif 'Triplet' in self.loss:
            self.margin = float(params.get('margin', 0.4))    # Margin used in loss function
        else:
            raise 'Unsupported loss function: {}'.format(self.loss)

        self.aug_mode = int(params.get('aug_mode', 1))    # Augmentation mode (1 is default)

        self.train_file = params.get('train_file')
        self.val_file = params.get('val_file', None)

        # Read model parameters
        if self.model_params_path is not None:
            self.model_params = ModelParams(self.model_params_path)
        else:
            self.model_params = None

        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

    def print(self):
        print('*Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e not in ['model_params']:
                print('{}: {}'.format(e, param_dict[e]))

        if self.model_params is not None:
            self.model_params.print()
        print('')

    def to_dict(self):
        param_dict = {}
        for e in param_dict:
            if e not in ['model_params']:
                param_dict[e] = param_dict[e]

        if self.model_params is not None:
            param_dict["model_params"] = self.model_params.to_dict()
