from environs import Env

from punctuator.src.path_manager import PathManager


class Config:
    EMBED_DIM = 256
    HIDDEN_DIM = 500
    BATCH_SIZE = 130
    EPOCH = 2
    LR = 0.1


class EnvFile:
    env = Env()
    env.read_env(str(PathManager.CREDENTIALS / '.env'))

    OPTIONS_FILE = env("OPTIONS_FILE")
    WEIGHT_FILE = env("WEIGHT_FILE")

    API_KEY = env('COMETML_API_KEY')
    PROJECT_NAME = env("COMETML_PROJECT_NAME")
    WORKSPACE = env("COMETML_WORKSPACE")
