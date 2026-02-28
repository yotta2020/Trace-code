from .style_poisoner import StylePoisoner

POISONERS = {
    "style": StylePoisoner,
}


def load_poisoner(cfg):
    return POISONERS[cfg.attacker.type.lower()](cfg)
