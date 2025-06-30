import os
import datetime

MODELS_DIRECTORY = lambda : os.environ["KONFAI_MODELS_DIRECTORY"]
CHECKPOINTS_DIRECTORY =lambda : os.environ["KONFAI_CHECKPOINTS_DIRECTORY"]
MODEL = lambda : os.environ["KONFAI_MODEL"]
PREDICTIONS_DIRECTORY =lambda : os.environ["KONFAI_PREDICTIONS_DIRECTORY"]
EVALUATIONS_DIRECTORY =lambda : os.environ["KONFAI_EVALUATIONS_DIRECTORY"]
STATISTICS_DIRECTORY = lambda : os.environ["KONFAI_STATISTICS_DIRECTORY"]
SETUPS_DIRECTORY = lambda : os.environ["KONFAI_SETUPS_DIRECTORY"]
CONFIG_FILE = lambda : os.environ["KONFAI_CONFIG_FILE"]
KONFAI_STATE = lambda : os.environ["KONFAI_STATE"]
KONFAI_ROOT = lambda : os.environ["KONFAI_ROOT"]
CUDA_VISIBLE_DEVICES = lambda : os.environ["CUDA_VISIBLE_DEVICES"]
KONFAI_NB_CORES = lambda : os.environ["KONFAI_NB_CORES"]
DATE = lambda : datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")