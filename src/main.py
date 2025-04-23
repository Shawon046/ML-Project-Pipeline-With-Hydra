import os
import sys
# # Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from pathlib import Path

import hydra
from omegaconf import DictConfig # type: ignore

from helper.misc_helper import *
from dataset.load_data import get_dataloader
from model.traditional_models import *
from model.accuracy_metrics import *
from dataset.load_nsl_data import *
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)  


def load_processed_dataloader(cfg: DictConfig) -> None:
    print(f"{cfg.run_type.capitalize()} {cfg.model} model on {cfg.dataset}")
    
    print(f"Starting point: {cfg.dataset_dir} ")
    train_dataloader = get_dataloader(cfg, True) 
    print(f'Datatype of train dataloader : {train_dataloader.datatype}')
    test_dataloader = get_dataloader(cfg, False) 
    return train_dataloader, test_dataloader

def load_processed_dataset(cfg: DictConfig) -> None:
    print(f"{cfg.run_type.capitalize()} {cfg.model} model on {cfg.dataset}")
    print(f"Starting point: {cfg.dataset_dir} ")

    X_train, y_train  = get_nsl_dataset(cfg, True) 
    X_test, y_test  = get_nsl_dataset(cfg, False) 
    return X_train, y_train, X_test, y_test 

def baseline_model_train_test(cfg, X_train, y_train, X_test, y_test):
    # Use this dataloaders to train and test models
    result_dict_test = baseline_traditional_ml_models(cfg, X_train, y_train, X_test, y_test)
    compare_accuracy_metrics(cfg, result_dict_test)


def llm_based_agent_train_test(cfg, X_train, y_train, X_test, y_test):
    print('Implement LLM based agent here')

@hydra.main(version_base=None, config_path="../conf", config_name="config")

def main(cfg: DictConfig):
    print(f'**** Main file ****')
    seed_everything(cfg)
    DEVICE = getDevice(cfg)
    check_cuda(cfg)
    # X_train, y_train, X_test, y_test = load_processed_dataset(cfg)
    # baseline_model_train_test(cfg, X_train, y_train, X_test, y_test)

    # llm_based_agent_train_test(cfg, X_train, y_train, X_test, y_test)
    print(f'**** End ****')

if __name__ == "__main__":
    main()