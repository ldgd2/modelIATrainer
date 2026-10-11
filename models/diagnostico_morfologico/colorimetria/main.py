import os
import sys
import argparse

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.logger import setup_logger
from common.config import load_yaml_config, merge_cli_args
from models.colorimetria.download.download_data import ensure_dataset
from models.colorimetria.dataset.dataset import build_dataloaders
from models.colorimetria.architecture.network import build_model
from models.colorimetria.trainer.trainer import ColorimetriaTrainer

logger = setup_logger("Colorimetria-Main")

def parse_args():
    parser = argparse.ArgumentParser(description="BellAI - Facial Colorimetry Model Trainer")
    default_config = os.path.join(os.path.dirname(__file__), "config", "config.yaml")

    parser.add_argument("--config", type=str, default=default_config, help="Path to YAML config")
    parser.add_argument("--dataset", type=str, dest="dataset_path", default=None, help="Path to dataset")
    parser.add_argument("--gdown_id", type=str, default=None, help="Google Drive file ID for dataset")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--drive_backup_dir", type=str, default=None, help="Google Drive backup folder")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of seasonal palettes")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda', 'cpu', 'auto')")

    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Initializing Colorimetria Training Module...")

    # 1. Load YAML and merge CLI overrides
    config = load_yaml_config(args.config)
    cli_overrides = {k: v for k, v in vars(args).items() if k not in ["config"] and v is not None}
    config = merge_cli_args(config, cli_overrides)

    logger.info(f"Loaded config: Batch Size={config.get('batch_size')}, Epochs={config.get('epochs')}, LR={config.get('learning_rate')}")

    # 2. Download / verify dataset
    dataset_dir = ensure_dataset(
        dataset_dir=config.get("dataset_path", "data/colorimetria"),
        gdown_id=config.get("gdown_id", "")
    )

    # 3. Build data loaders
    train_loader, val_loader, detected_classes = build_dataloaders(
        data_dir=dataset_dir,
        batch_size=int(config.get("batch_size", 32)),
        img_size=int(config.get("img_size", 224)),
        val_split=float(config.get("val_split", 0.15)),
        num_workers=int(config.get("num_workers", 2)),
        num_classes=int(config.get("num_classes", 4))
    )

    # 4. Build Model
    model = build_model(
        num_classes=detected_classes,
        pretrained=True
    )

    # 5. Train & Sync
    trainer = ColorimetriaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    trainer.fit()

if __name__ == "__main__":
    main()
