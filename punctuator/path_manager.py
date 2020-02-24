from pathlib import Path


class PathManager:
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    DATA: Path = BASE_DIR / "data"

    ROOT_DIR: Path = BASE_DIR / 'punctuator'
    TESTS: Path = ROOT_DIR / "tests"
    SRC: Path = ROOT_DIR / "src"
    CREDENTIALS: Path = ROOT_DIR / "credentials"

    # data directory
    RAW: Path = DATA / "raw"
    INTERIM: Path = DATA / "interim"
    PROCESSED: Path = DATA / "processed"