from pathlib import Path
from typing import List


class PathManager:
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    DATA: Path = BASE_DIR / "data"
    NOTEBOOKS: Path = BASE_DIR / "notebooks"
    MODULES: Path = BASE_DIR / "modules"
    TESTS: Path = BASE_DIR / "tests"
    SRC: Path = BASE_DIR / "src"

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

