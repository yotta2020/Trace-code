import torch
import torch.nn as nn
from typing import List, Optional
from .victim import BaseVictim
from .codebert.defect import Victim as DefectCodeBERTVictim
from .codebert.clone import Victim as CloneCoedBERTVictim
from .codebert.refine import Victim as RefineCodeBERTVictim
from .codet5.defect import Victim as DefectCodeT5Victim
from .codet5.clone import Victim as CloneCodeT5Victim
from .codet5.refine import Victim as RefineCodeT5Victim
from .starcoder.defect import Victim as DefectStarCoderVictim
from .starcoder.clone import Victim as CloneStarCoderVictim
from .starcoder.refine import Victim as RefineStarCoderVictim


Victim_List = {
    "codebert": {
        "defect": DefectCodeBERTVictim,
        "clone": CloneCoedBERTVictim,
        "refinement": RefineCodeBERTVictim,
        "refine": RefineCodeBERTVictim,  # Alias
    },
    "codet5": {
        "defect": DefectCodeT5Victim,
        "clone": CloneCodeT5Victim,
        "refinement": RefineCodeT5Victim,
        "refine": RefineCodeT5Victim,  # Alias
    },
    "starcoder": {
        "defect": DefectStarCoderVictim,
        "clone": CloneStarCoderVictim,
        "refinement": RefineStarCoderVictim,
        "refine": RefineStarCoderVictim,  # Alias
    },
}


def load_victim(cfg):
    victim = Victim_List[cfg.victim.type][cfg.task.lower()](cfg)
    return victim
