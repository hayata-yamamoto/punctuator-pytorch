from environs import Env

from punctuator.src.path_manager import PathManager


class Config:
    EMBED_DIM = 256
    HIDDEN_DIM = 500
    BATCH_SIZE = 128
    EPOCH = 10
    LR = 0.01


class EnvFile:
    env = Env()
    env.read_env(str(PathManager.CREDENTIALS / '.env'))

    OPTIONS_FILE = env("OPTIONS_FILE")
    WEIGHT_FILE = env("WEIGHT_FILE")
