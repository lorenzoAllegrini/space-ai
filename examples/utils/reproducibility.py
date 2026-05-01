"""Reproducibility utilities for SpaceAI experiments."""

import random
import os
import numpy as np
import torch
import logging

def set_seed(seed: int = 42):
    """
    Sets the seed for various random number generators to ensure 
    that experiments are reproducible across different runs.
    """
    if seed is None:
        return
        
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass
            
    logging.debug(f"Global seed set to {seed} for reproducibility.")
