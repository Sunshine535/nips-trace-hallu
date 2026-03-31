"""Load trace_config.yaml and merge with argparse defaults."""

import yaml
from pathlib import Path
from typing import Optional


def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


_SECTION_MAP = {
    "collect_traces":            "generator",
    "generate_traces":           "generator",
    "train_onset_detector":      "detector",
    "train_intervention_policy": "intervention",
    "train_policy_online":       "intervention",
    "eval_chi":                  "eval",
    "eval_intervention":         "eval",
}


def load_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        config_path = "configs/trace_config.yaml"
    p = Path(config_path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def apply_config_defaults(args, script_name: str, config: dict) -> None:
    """Overwrite argparse *defaults* with config values (CLI flags win)."""
    section_key = _SECTION_MAP.get(script_name)
    if section_key is None:
        return

    section = config.get(section_key, {})
    if not isinstance(section, dict):
        return

    flat = _flatten(section)

    arg_dict = vars(args)
    for key, value in flat.items():
        leaf = key.rsplit(".", 1)[-1]
        if leaf in arg_dict and arg_dict[leaf] is None:
            arg_dict[leaf] = value
