# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom SpeciesNet server that draws and saves annotated images.

Extends the standard SpeciesNet server to:
1. Run inference on input images
2. Draw bounding boxes on detected objects
3. Save annotated images alongside originals with '_annotated' suffix
4. Return annotated image path in prediction response
"""

from pathlib import Path
from typing import Optional

from absl import app
from absl import flags
from fastapi import HTTPException
import litserve as ls
from PIL import Image

from speciesnet import DEFAULT_MODEL
from speciesnet import draw_bboxes
from speciesnet import file_exists
from speciesnet import SpeciesNet

_PORT = flags.DEFINE_integer(
    "port",
    8000,
    "Port to run the server on.",
)
_API_PATH = flags.DEFINE_string(
    "api_path",
    "/predict",
    "URL path for the server endpoint.",
)
_WORKERS_PER_DEVICE = flags.DEFINE_integer(
    "workers_per_device",
    1,
    "Number of server replicas per device.",
)
_TIMEOUT = flags.DEFINE_integer(
    "timeout",
    30,
    "Timeout (in seconds) for requests.",
)
_BACKLOG = flags.DEFINE_integer(
    "backlog",
    2048,
    "Maximum number of connections to hold in backlog.",
)
_MODEL = flags.DEFINE_string(
    "model",
    DEFAULT_MODEL,
    "SpeciesNet model to load.",
)
_GEOFENCE = flags.DEFINE_bool(
    "geofence",
    True,
    "Whether to enable geofencing or not.",
)
_EXTRA_FIELDS = flags.DEFINE_list(
    "extra_fields",
    None,
    "Comma-separated list of extra fields to propagate from request to response.",
)
_SAVE_ANNOTATED = flags.DEFINE_bool(
    "save_annotated",
    True,
    "Whether to save annotated images with bounding boxes.",
)


def save_annotated_image(filepath: str, detections: list[dict]) -> Optional[str]:
    """Draw bounding boxes and save annotated image alongside original.

    Args:
        filepath: Path to the original image.
        detections: List of detection dictionaries with bbox info.

    Returns:
        Path to the saved annotated image, or None if no detections.
    """
    if not detections:
        return None

    try:
        img = Image.open(filepath).convert("RGB")
        annotated_img = draw_bboxes(img, detections)
        # Convert RGBA back to RGB for JPEG saving
        annotated_img = annotated_img.convert("RGB")

        # Generate annotated filename
        path = Path(filepath)
        annotated_path = path.parent / f"{path.stem}_annotated{path.suffix}"

        # Save with high quality
        annotated_img.save(str(annotated_path), quality=90)

        return str(annotated_path)
    except Exception as e:
        print(f"Error saving annotated image for {filepath}: {e}")
        return None


class AnnotatingSpeciesNetAPI(ls.LitAPI):
    """Extended SpeciesNet API that saves annotated images.

    This class extends the standard SpeciesNet server to also draw bounding
    boxes on detected objects and save the annotated images alongside the
    original files.
    """

    def __init__(
        self,
        model_name: str,
        geofence: bool = True,
        extra_fields: Optional[list[str]] = None,
        save_annotated: bool = True,
        api_path: str = "/predict",
    ) -> None:
        """Initializes the annotating SpeciesNet API server.

        Args:
            model_name:
                String value identifying the model to be loaded.
            geofence:
                Whether to enable geofencing or not. Defaults to `True`.
            extra_fields:
                Comma-separated list of extra fields to propagate.
            save_annotated:
                Whether to save annotated images. Defaults to `True`.
            api_path:
                URL path for the server endpoint. Defaults to `/predict`.
        """
        super().__init__()
        self.api_path = api_path
        self.model_name = model_name
        self.geofence = geofence
        self.extra_fields = extra_fields or []
        self.save_annotated = save_annotated

    def setup(self, device):
        del device  # Unused.
        self.model = SpeciesNet(self.model_name, geofence=self.geofence)

    def decode_request(self, request, context):
        del context  # Unused.
        for instance in request["instances"]:
            filepath = instance["filepath"]
            if not file_exists(filepath):
                raise HTTPException(400, f"Cannot access filepath: `{filepath}`")
        return request

    def _propagate_extra_fields(
        self, instances_dict: dict, predictions_dict: dict
    ) -> dict:
        predictions = predictions_dict["predictions"]
        new_predictions = {p["filepath"]: p for p in predictions}
        for instance in instances_dict["instances"]:
            for field in self.extra_fields:
                if field in instance:
                    new_predictions[instance["filepath"]][field] = instance[field]
        return {"predictions": list(new_predictions.values())}

    def predict(self, instances_dict, context):
        del context  # Unused.
        predictions_dict = self.model.predict(instances_dict=instances_dict)
        assert predictions_dict is not None

        result = self._propagate_extra_fields(instances_dict, predictions_dict)

        # Draw and save annotated images if enabled
        if self.save_annotated:
            for prediction in result["predictions"]:
                filepath = prediction["filepath"]
                detections = prediction.get("detections", [])
                annotated_path = save_annotated_image(filepath, detections)
                prediction["annotated_filepath"] = annotated_path

        return result

    def encode_response(self, output, context):
        del context  # Unused.
        return output


def main(argv: list[str]) -> None:
    del argv  # Unused.

    api = AnnotatingSpeciesNetAPI(
        model_name=_MODEL.value,
        geofence=_GEOFENCE.value,
        extra_fields=_EXTRA_FIELDS.value,
        save_annotated=_SAVE_ANNOTATED.value,
        api_path=_API_PATH.value,
    )
    server = ls.LitServer(
        api,
        accelerator="auto",
        devices="auto",
        workers_per_device=_WORKERS_PER_DEVICE.value,
        timeout=_TIMEOUT.value,
    )
    server.run(
        port=_PORT.value,
        generate_client_file=False,
        backlog=_BACKLOG.value,
    )


if __name__ == "__main__":
    app.run(main)
