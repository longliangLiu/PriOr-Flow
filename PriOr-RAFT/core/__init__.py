import torch
import numpy as np
from omegaconf import OmegaConf, ListConfig

def fetch_dataloader(cfg, logger):
    if cfg.type == '360flow':
        from .datasets import Flow360, FlowScape, MPFDataset, OmniFlowNet_Dataset, OmniPhotos
        __all__ = {
            'Flow360': Flow360,
            'FlowScape': FlowScape,
            'MPFDataset': MPFDataset,
            'OmniFlowNet_Dataset': OmniFlowNet_Dataset, 
            'OmniPhotos': OmniPhotos
        }

        all_dataset = {}
        for dataset_name, scenes in zip(cfg.name, cfg.scene):
            if scenes is None:
                dataset = __all__[dataset_name](aug_params=cfg.aug_params, split=cfg.split)
                logger.info(f'Reading {len(dataset)} samples from {dataset_name}.')
                all_dataset[dataset_name] = dataset
            else:
                if not OmegaConf.is_list(scenes):
                    scenes = [scenes]
                for scene in scenes:
                    dataset = __all__[dataset_name](aug_params=cfg.aug_params, scene=scene, split=cfg.split)
                    logger.info(f'Reading {len(dataset)} samples from {dataset_name}.')
                    all_dataset[dataset_name + '_' + scene] = dataset
            
        if cfg.split == 'train':
            dataset = None
            for key, value in all_dataset.items():
                dataset = dataset + value if dataset is not None else value
            logger.info(f'----------------------------------------------------------')
            logger.info(f'Adding {len(dataset)} samples from all datasets for {cfg.split}.')
            logger.info(f'----------------------------------------------------------')
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size // cfg.num_gpu, num_workers=8, pin_memory=True, shuffle=True,drop_last=True)
            return dataloader
        else:
            all_dataloader = {}
            for key, value in all_dataset.items():
                all_dataloader[key] = torch.utils.data.DataLoader(value, batch_size=1, num_workers=8, pin_memory=True, shuffle=False, drop_last=False)
            return all_dataloader
            