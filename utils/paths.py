import os

# root project and weights folder
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(PROJECT_PATH, "weights")

# Dataset paths
COIN_PATH = os.environ.get("DROPD_TW_COIN_PATH", os.path.join(PROJECT_PATH, "data", "COIN"))
CT_PATH = None
YC_PATH = None
ARA_PATH = None
MR_PATH = None
