import os

ROOT_FOLDER = os.path.join(os.path.realpath(os.path.dirname(__file__)))

DATA_FOLDER = f'{ROOT_FOLDER}/data'
TRAIN_DATA_PATH = f'{DATA_FOLDER}/tmp'
TEST_DATA_PATH = f'{DATA_FOLDER}/test'
CHECKPOINT_PATH = f'{ROOT_FOLDER}/checkpoint/model.h5'

LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
INPUT_SHAPE = (50, 150, 1)
OUTPUT_SIZE = len(LABELS)
