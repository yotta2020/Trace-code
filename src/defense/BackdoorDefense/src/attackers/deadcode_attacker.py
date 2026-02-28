from typing import *
from src.defense.BackdoorDefense.src.utils import logger
from .attacker import Attacker


class DeadCodeAttacker(Attacker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("load DeadCodeAttacker")
