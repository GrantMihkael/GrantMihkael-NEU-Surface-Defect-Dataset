"""Submission-ready metal defect classification package."""

from .utils import load_config, save_json, set_seed
from .trainer import train_model
from .evaluation import evaluate_model

__all__ = ["load_config", "save_json", "set_seed", "train_model", "evaluate_model"]
