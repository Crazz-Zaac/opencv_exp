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
    num_epochs: int
    device: str = "cpu"
    log_steps: int = 100
    save_steps: int = 1000
    model_dir: Path = Path("models")
    model_name: str = "model"
    model_extension: str = "pt"
    log_dir: Path = Path("logs")
    log_name: str = "log"
    log_extension: str = "csv"
    seed: int = 42
    verbose: bool = True

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
        
        # if res_dir does not exist, create it
        if not config.rslt_dir.exists():
            config.rslt_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = config.rslt_dir / config.model_dir
        
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = config.rslt_dir / config.log_dir
        
        