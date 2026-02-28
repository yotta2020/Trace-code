from .defender import Defender
from .onion_defender import ONIONDefender

DEFENDERS = {
    "base": Defender,
    "onion": ONIONDefender,
}


def load_defender(cfg):
    return DEFENDERS[cfg.defender.type.lower()](cfg)
