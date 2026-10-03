import os
import time

import cv2
import insightface
import numpy as np
import onnxruntime as ort

# Containers often expose every host core while enforcing a small CPU
# quota, which makes default thread pools thrash. Cap them.
CPU_THREADS = int(os.getenv("CPU_THREADS", "2"))
cv2.setNumThreads(CPU_THREADS)


class FaceEngine:
    """
    InsightFace-based face detection and embedding engine.

    Responsibilities:
        1. Load InsightFace.
        2. Detect faces.
        3. Generate 512-dimensional face embeddings.
    """

    def __init__(self):

        print("=" * 60)
        print("INITIALIZING FACE ENGINE")
        print("=" * 60)

        # ----------------------------------------------------
        # Load InsightFace
        # ----------------------------------------------------

        # Only detection and recognition are used. Skipping the
        # landmark and gender/age models lowers memory use.
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = CPU_THREADS
        session_options.inter_op_num_threads = 1

        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            sess_options=session_options,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )

        # CPU execution.
        #
        # Later, the deployment layer can choose an
        # appropriate execution provider depending on the
        # customer's hardware.
        self.app.prepare(
            ctx_id=-1,
            det_size=(640, 640)
        )

        print("InsightFace model loaded.")
        print("=" * 60)


    # ========================================================
    # FACE DETECTION
    # ========================================================

    def detect_faces(self, frame):
        """
        Detect all faces in an image/frame.

        Returns:
            list of InsightFace Face objects
        """

        if frame is None:
            return []

        started = time.perf_counter()

        faces = self.app.get(frame)

        print(
            f"[timing] face detect+embed: "
            f"{time.perf_counter() - started:.2f}s"
        )

        return faces


    # ========================================================
    # SINGLE FACE
    # ========================================================

    def get_single_face(self, frame):
        """
        Return a face only when exactly one face is detected.

        Returns:
            InsightFace Face object
            or None
        """

        faces = self.detect_faces(frame)

        if len(faces) != 1:
            return None

        return faces[0]


    # ========================================================
    # EMBEDDING
    # ========================================================

    def get_embedding(self, face):
        """
        Extract a 512-dimensional normalized embedding
        from an InsightFace Face object.
        """

        if face is None:
            return None

        embedding = getattr(
            face,
            "normed_embedding",
            None
        )

        if embedding is None:
            return None

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Validate embedding
        # ----------------------------------------------------

        if embedding.ndim != 1:
            raise ValueError(
                "Face embedding must be a 1D vector."
            )

        if embedding.shape[0] != 512:
            raise ValueError(
                f"Expected 512-dimensional embedding, "
                f"got {embedding.shape[0]}."
            )

        # ----------------------------------------------------
        # Make a standalone copy
        # ----------------------------------------------------

        embedding = embedding.copy()

        return embedding


    # ========================================================
    # FACE + EMBEDDING
    # ========================================================

    def analyze(self, frame):
        """
        Detect faces and generate embeddings.

        Returns a list of dictionaries:

        [
            {
                "face": <InsightFace Face>,
                "embedding": <512-D numpy array>
            }
        ]
        """

        faces = self.detect_faces(frame)

        results = []

        for face in faces:

            embedding = self.get_embedding(
                face
            )

            if embedding is None:
                continue

            results.append(
                {
                    "face": face,
                    "embedding": embedding,
                }
            )

        return results


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("Testing Face Engine...")
    print()

    engine = FaceEngine()

    print()
    print("Face Engine initialized successfully.")
    print()