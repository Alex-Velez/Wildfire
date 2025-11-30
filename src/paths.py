import os
from pathlib import Path
from tabulate import tabulate

BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
SRC_DIR: Path = BASE_DIR / "src"
CONFIG_DIR: Path = BASE_DIR / "config"
PARAMS_DIR: Path = BASE_DIR / "parameters"

def short_path(path: Path) -> str:
    return os.sep.join([path.parts[0], "...", os.sep.join(path.parts[-2:])])

if __name__ == "__main__":
    info_table = tabulate(
        [
            ["Global Variable", "Absolute Path"],
            ["BASE_DIR", f"{short_path(BASE_DIR)}"],
            ["DATA_DIR", f"{short_path(DATA_DIR)}"],
            ["SRC_DIR", f"{short_path(SRC_DIR)}"],
            ["CONFIG_DIR", f"{short_path(CONFIG_DIR)}"],
            ["PARAMS_DIR", f"{short_path(PARAMS_DIR)}"],
        ],
        headers="firstrow",
        tablefmt="simple_grid",
        numalign="center",
        stralign="center",
    )
    print(info_table)