import os
import sys
import time
import cv2
import numpy as np
import torch


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

OFFICIAL_REPO = os.path.join(
    PROJECT_ROOT,
    "models",
    "minifasnet",
    "Silent-Face-Anti-Spoofing"
)

MODEL_DIR = os.path.join(
    OFFICIAL_REPO,
    "resources",
    "anti_spoof_models"
)


# ============================================================
# IMPORT OFFICIAL MINIFASNET CODE
# ============================================================

if OFFICIAL_REPO not in sys.path:
    sys.path.insert(0, OFFICIAL_REPO)

from src.model_lib.MiniFASNet import (
    MiniFASNetV2,
    MiniFASNetV1SE,
)

from src.generate_patches import CropImage
from src.data_io.transform import ToTensor


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIGS = [
    {
        "name": "2.7_80x80_MiniFASNetV2.pth",
        "scale": 2.7,
        "model_type": "MiniFASNetV2",
    },
    {
        "name": "4_0_0_80x80_MiniFASNetV1SE.pth",
        "scale": 4.0,
        "model_type": "MiniFASNetV1SE",
    },
]


# ============================================================
# PAD ENGINE
# ============================================================

class PAD:

    def __init__(self, device="cpu"):

        # Cap torch threads; see CPU_THREADS note in face_engine.py.
        torch.set_num_threads(int(os.getenv("CPU_THREADS", "2")))

        self.device = torch.device(device)

        print("=" * 60)
        print("INITIALIZING MINI FASNET PAD")
        print("=" * 60)

        print(f"Device: {self.device}")
        print(f"Model directory: {MODEL_DIR}")

        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(
                f"Model directory not found:\n{MODEL_DIR}"
            )

        self.cropper = CropImage()
        self.to_tensor = ToTensor()

        self.models = []

        self._load_models()

        print("=" * 60)
        print("MINIFASNET PAD READY")
        print("=" * 60)


    # ========================================================
    # LOAD OFFICIAL MODELS
    # ========================================================

    def _load_models(self):

        for config in MODEL_CONFIGS:

            model_path = os.path.join(
                MODEL_DIR,
                config["name"]
            )

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model weight not found:\n{model_path}"
                )

            print()
            print(f"Loading: {config['name']}")

            # ------------------------------------------------
            # Create exact official architecture
            # ------------------------------------------------

            if config["model_type"] == "MiniFASNetV2":

                model = MiniFASNetV2(
                    embedding_size=128,
                    conv6_kernel=(5, 5),
                    drop_p=0.2,
                    num_classes=3,
                    img_channel=3,
                )

            elif config["model_type"] == "MiniFASNetV1SE":

                model = MiniFASNetV1SE(
                    embedding_size=128,
                    conv6_kernel=(5, 5),
                    drop_p=0.75,
                    num_classes=3,
                    img_channel=3,
                )

            else:
                raise ValueError(
                    f"Unknown model type: {config['model_type']}"
                )

            # ------------------------------------------------
            # Load checkpoint
            # ------------------------------------------------

            checkpoint = torch.load(
                model_path,
                map_location=self.device
            )

            # Some checkpoints may contain "module." prefixes.
            state_dict = {}

            for key, value in checkpoint.items():

                if key.startswith("module."):
                    key = key[7:]

                state_dict[key] = value

            model.load_state_dict(
                state_dict,
                strict=True
            )

            model.to(self.device)
            model.eval()

            self.models.append(
                {
                    "name": config["name"],
                    "scale": config["scale"],
                    "model": model,
                }
            )

            print("Loaded successfully.")


    # ========================================================
    # CROP FACE USING OFFICIAL MINIFASNET CROP LOGIC
    # ========================================================

    def _crop_face(
        self,
        frame,
        bbox,
        scale
    ):

        # MiniFASNet expects:
        # [x, y, width, height]

        image = self.cropper.crop(
            org_img=frame,
            bbox=bbox,
            scale=scale,
            out_w=80,
            out_h=80,
            crop=True,
        )

        return image


    # ========================================================
    # PREPROCESS
    # ========================================================

    def _preprocess(self, image):

        # Official repository ToTensor:
        # H x W x C
        # [0,255]
        # ->
        # C x H x W
        # [0,1]

        tensor = self.to_tensor(image)

        if not torch.is_tensor(tensor):
            tensor = torch.tensor(tensor)

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(
            self.device,
            dtype=torch.float32
        )

        return tensor


    # ========================================================
    # SINGLE MODEL PREDICTION
    # ========================================================

    def _predict_model(
        self,
        model,
        image
    ):

        started = time.perf_counter()

        tensor = self._preprocess(image)

        with torch.no_grad():

            output = model(tensor)

            probability = torch.softmax(
                output,
                dim=1
            )

        print(
            f"[timing] PAD model: "
            f"{time.perf_counter() - started:.2f}s"
        )

        return probability.cpu().numpy()[0]


    # ========================================================
    # MAIN PAD CHECK
    # ========================================================

    def check(
        self,
        frame,
        bbox
    ):

        """
        Parameters
        ----------
        frame:
            OpenCV BGR image.

        bbox:
            [x, y, width, height]

        Returns
        -------
        dict
        """

        if frame is None:
            return {
                "label": "UNKNOWN",
                "confidence": 0.0,
                "class_0": 0.0,
                "class_1": 0.0,
                "class_2": 0.0,
            }

        if bbox is None:
            return {
                "label": "UNKNOWN",
                "confidence": 0.0,
                "class_0": 0.0,
                "class_1": 0.0,
                "class_2": 0.0,
            }

        # ----------------------------------------------------
        # Official fusion
        #
        # prediction =
        # V2 prediction + V1SE prediction
        # ----------------------------------------------------

        fused_prediction = np.zeros(
            3,
            dtype=np.float32
        )

        for model_info in self.models:

            cropped = self._crop_face(
                frame,
                bbox,
                model_info["scale"]
            )

            prediction = self._predict_model(
                model_info["model"],
                cropped
            )

            fused_prediction += prediction

        # ----------------------------------------------------
        # Official fusion decision
        # ----------------------------------------------------

        predicted_class = int(
            np.argmax(fused_prediction)
        )

        # There are two models, therefore divide the summed
        # probability by 2 to obtain the fused average score.

        probabilities = fused_prediction / len(self.models)

        confidence = float(
            probabilities[predicted_class]
        )

        # Official MiniFASNet convention:
        #
        # class 1 = REAL
        # class 0 = SPOOF
        # class 2 = SPOOF

        if predicted_class == 1:

            label = "REAL"

        else:

            label = "SPOOF"

        return {
            "label": label,
            "confidence": confidence,
            "class_0": float(probabilities[0]),
            "class_1": float(probabilities[1]),
            "class_2": float(probabilities[2]),
        }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("Testing PAD engine...")
    print()

    pad = PAD()

    print()
    print("PAD engine loaded successfully.")
    print()