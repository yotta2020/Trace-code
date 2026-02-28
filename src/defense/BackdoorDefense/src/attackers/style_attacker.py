from typing import *
from src.defense.BackdoorDefense.src.utils import logger
from .attacker import Attacker


class StyleAttacker(Attacker):
    def __init__(self, cfg):
        super().__init__(cfg)
        logger.info("load StyleAttacker")
