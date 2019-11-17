from dataclasses import dataclass

from environs import Env

from punctuator.src.path_manager import PathManager


@dataclass(frozen=True)
class Config:
    EMBED_DIM = 256
    HIDDEN_DIM = 500
    BATCH_SIZE = 130
    EPOCH = 100
    LR = 0.1

    class CometMl:
        env = Env()
        env.read_env(PathManager.CREDENTIALS / '.env')
        API_KEY = env('COMETML_API_KEY')
        PROJECT_NAME = env("COMETML_PROJECT_NAME")
        WORKSPACE = env("COMETML_WORKSPACE")
