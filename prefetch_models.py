"""Pre-download uniface's ONNX model weights, run once at build time.

engine.py used to let uniface fetch these from GitHub lazily on first import
(FaceEngine.__init__), so every cold start depended on GitHub being
reachable. GitHub intermittently closed the connection mid-download
(RemoteDisconnected), which crashed boot with
`ConnectionError: Download failed for 'scrfd_500m' after 3 attempts`.

Running this during the build/image step bakes the weights into uniface's
cache dir (~/.uniface/models) so startup loads them from local disk and
never touches the network. Both SCRFD variants are fetched regardless of
the current DET_MODEL setting, since that can be changed via env var at
runtime without triggering a rebuild.
"""

from uniface.constants import ArcFaceWeights, SCRFDWeights
from uniface.model_store import download_models

MODELS = [
    SCRFDWeights.SCRFD_500M_KPS,
    SCRFDWeights.SCRFD_10G_KPS,
    ArcFaceWeights.MNET,
]

if __name__ == "__main__":
    download_models(MODELS, max_retries=5, timeout=120)
