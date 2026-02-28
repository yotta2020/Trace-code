from .attacker import Attacker
from .deadcode_attacker import DeadCodeAttacker
from .style_attacker import StyleAttacker

ATTACKERS = {"base": Attacker, "deadcode": DeadCodeAttacker, "style": StyleAttacker}


def load_attacker(cfg):
    return ATTACKERS[cfg.attacker.type](cfg)
