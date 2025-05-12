"""YOLO training and evaluation module."""

from pathlib import Path
import torch
from ultralytics import YOLO
from roboflow import Roboflow


class YOLOTrainer:
    """Class to handle YOLO model training and evaluation."""

    def __init__(self, base_dir: str | Path, output_dir: str | Path):
        """Initialize the YOLO trainer.

        Args:
            base_dir: Base directory for the project
            output_dir: Directory for model outputs
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)

        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory {self.base_dir} does not exist")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_roboflow(
        self,
        workspace_id: str,
        project_id: str,
        api_key: str,
        project_version: int,
        model_type: str = "yolov11",
    ) -> str:
        """Set up and download dataset from Roboflow.

        Args:
            workspace_id: Roboflow workspace ID
            project_id: Roboflow project ID
            api_key: Roboflow API key
            project_version: Project version number
            model_type: Type of YOLO model to use

        Returns:
            Path to the downloaded dataset
        """
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace_id).project(project_id)
        version = project.version(project_version)
        dataset = version.download(model_type)
        return dataset.location

    def train(
        self,
        dataset_path: str | Path,
        model_type: str = "yolo11s-seg.pt",
        epochs: int = 100,
        img_size: int = 640,
        batch_size: int | None = None,
    ):
        """Train the YOLO model.

        Args:
            dataset_path: Path to the dataset
            model_type: Type of YOLO model to use
            epochs: Number of training epochs
            img_size: Input image size
            batch_size: Batch size (if None, will be scaled with GPU count)
        """
        # Setup device
        gpu_count = torch.cuda.device_count()
        device_str = (
            ",".join(str(i) for i in range(gpu_count)) if gpu_count > 0 else "cpu"
        )
        print(f"🧠 Training on: {device_str} ({gpu_count} GPU(s) detected)")

        # Initialize model
        model = YOLO(model_type)

        # Set batch size based on GPU count if not specified
        if batch_size is None:
            batch_size = 8 * max(gpu_count, 1)

        # Train model
        results = model.train(
            data=str(Path(dataset_path) / "data.yaml"),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            workers=2,
            device=device_str,
            fliplr=0.5,
            mosaic=1.0,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            perspective=0.001,
            project=str(self.output_dir / "runs/segment"),
            name="train",
        )
        return results

    def evaluate(self, model_path: str | Path, dataset_path: str | Path):
        """Evaluate the trained model.

        Args:
            model_path: Path to the trained model weights
            dataset_path: Path to the dataset
        """
        model = YOLO(model_path)
        metrics = model.val(split="test", data=str(Path(dataset_path) / "data.yaml"))
        return metrics


# Example usage:
"""
# trainer = YOLOTrainer(base_dir='/path/to/base', output_dir='/path/to/output')

# Download dataset
# dataset_path = trainer.setup_roboflow(
#     workspace_id="your_workspace",
#     project_id="your_project",
#     api_key="your_api_key",
#     project_version=1
# )

# Train model
# results = trainer.train(dataset_path)

# Evaluate model
# metrics = trainer.evaluate(
#     model_path='path/to/best.pt',
#     dataset_path=dataset_path
# )
"""
