from pathlib import Path
from typing import List


class PathManager:
    BASE_DIR: Path = Path(__file__).resolve().parents[3]
    DATA: Path = BASE_DIR / "data"
    
    ROOT_DIR: Path = BASE_DIR / 'punctuator'
    NOTEBOOKS: Path = ROOT_DIR / "notebooks"
    MODULES: Path = ROOT_DIR / "modules"
    TESTS: Path = ROOT_DIR / "tests"
    SRC: Path = ROOT_DIR / "src"

    # data directory
    RAW: Path = DATA / "raw"
    INTERIM: Path = DATA / "interim"
    PROCESSED: Path = DATA / "processed"

    # notebooks directory
    EXPLORATORY: Path = NOTEBOOKS / "exploratory"
    PREDICTIVE: Path = NOTEBOOKS / "predictive"

    # project directory
    CORE: Path = SRC / "core"
    DATASETS: Path = SRC / "datasets"
    FEATURES: Path = SRC / "features"
    MODELS: Path = SRC / "models"
    TASKS: Path = SRC / "tasks"

