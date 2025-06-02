import os
import datetime

MODELS_DIRECTORY = lambda : os.environ["DL_API_MODELS_DIRECTORY"]
CHECKPOINTS_DIRECTORY =lambda : os.environ["DL_API_CHECKPOINTS_DIRECTORY"]
MODEL = lambda : os.environ["DL_API_MODEL"]
PREDICTIONS_DIRECTORY =lambda : os.environ["DL_API_PREDICTIONS_DIRECTORY"]
EVALUATIONS_DIRECTORY =lambda : os.environ["DL_API_EVALUATIONS_DIRECTORY"]
STATISTICS_DIRECTORY = lambda : os.environ["DL_API_STATISTICS_DIRECTORY"]
SETUPS_DIRECTORY = lambda : os.environ["DL_API_SETUPS_DIRECTORY"]
CONFIG_FILE = lambda : os.environ["DEEP_LEARNING_API_CONFIG_FILE"]
DL_API_STATE = lambda : os.environ["DL_API_STATE"]
DEEP_LEARNING_API_ROOT = lambda : os.environ["DEEP_LEARNING_API_ROOT"]
CUDA_VISIBLE_DEVICES = lambda : os.environ["CUDA_VISIBLE_DEVICES"]

DATE = lambda : datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")