from pydantic import BaseModel
from pathlib import Path
from typing import List, Tuple, Optional
from enum import Enum
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler



class TrainerConfig(BaseModel):
    rslt_dir: Path
    expt_name: str
    run_name: str
    log_every: int
    chkpt_every: int
    best_model_name: str
    device: str = "cpu"

    class Config:
        arbitrary_types_allowed = True
        

class Trainer:
    def __init__(
        self,
        model,
        config: TrainerConfig,
        optimizer: Optimizer,
        criterion,
        scheduler: Optional[_LRScheduler] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config
        
        # if res_dir does not exist, create it inside res_dir/expt_name
        if not config.rslt_dir.exists():
            config.rslt_dir.mkdir(parents=True, exist_ok=True)
        self.expt_dir = config.rslt_dir / config.expt_name
        if not self.expt_dir.exists():
            self.expt_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.expt_dir / "models"        
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.expt_dir / "logs"
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.model_dir / config.best_model_name
        
        self.log_file = self.log_dir / f"{config.run_name}.log"
        self.log_file.touch()
        
        
    def train(self, train_loader, val_loader, num_epochs):
        pass
    
    def save_model(self, epoch, val_loss):
        pass
    
    def load_model(self, model_path):
        pass
    
    def log(self, message):
        with open(self.log_file, "a") as f:
            f.write(f"{message}\n")
            print(message)