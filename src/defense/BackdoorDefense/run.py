import sys
import os
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import hydra
import json
import pandas as pd
from src.defense.BackdoorDefense.src.data import load_dataset
from src.defense.BackdoorDefense.src.victims import load_victim
from src.defense.BackdoorDefense.src.attackers import load_attacker
from src.defense.BackdoorDefense.src.defenders import load_defender
from src.defense.BackdoorDefense.src.utils import logger, set_seed


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg):
    # print(omegaconf.OmegaConf.to_yaml(cfg))
    cfg.attacker.poisoner.triggers = [str(_) for _ in cfg.attacker.poisoner.triggers]

    logger.info(f"task = {cfg.task}")
    logger.info(f"poison_rate = {cfg.victim.poison_rate}")
    logger.info(f"triggers = {cfg.attacker.poisoner.triggers}")
    logger.info(f"defender = {cfg.defender.type}")

    set_seed(cfg.seed)
    victim = load_victim(cfg)
    attacker = load_attacker(cfg)
    defender = load_defender(cfg)
    dataset = load_dataset(cfg)
    # dataset["test"] = random.sample(dataset["test"], 300)
    res_dict = attacker.eval(victim, dataset, defender)

    defense_type = cfg.common.defense_type[cfg.defender.type]
    if defense_type == "detect":
        # Save to old results path for compatibility
        output_dir = "results"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = (
            Path(output_dir)
            / f"{cfg.task}_{cfg.victim.type}_{cfg.defender.type}_{cfg.victim.poison_rate}_{cfg.attacker.poisoner.triggers[0]}.json"
        )
        path.write_text(json.dumps(res_dict, indent=4))

        # Save defense results to new structured path
        if hasattr(defender, 'defense_results'):
            # Build path: results/defense/backdoordefense/{defender}/{task}/{model}/{attack_way}_{poisoned_rate}.json
            attack_way = cfg.attacker.poisoner.triggers[0]
            poisoned_rate = cfg.victim.poison_rate
            model_name = cfg.victim.type.lower()

            results_path = project_root / "results" / "defense" / "backdoordefense" / cfg.defender.type / cfg.task / model_name
            results_path.mkdir(parents=True, exist_ok=True)

            results_file = results_path / f"{attack_way}_{poisoned_rate}.json"

            logger.info(f"Saving defense results to: {results_file}")
            results_file.write_text(json.dumps(defender.defense_results, indent=4))
            logger.info(f"Defense results saved successfully!")


if __name__ == "__main__":
    main()
