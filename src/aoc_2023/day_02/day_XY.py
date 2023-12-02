from pathlib import Path
from typing import Union, List


def read_file(file_path: Union[str | Path]) -> List[str]:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    assert file_path.is_file(), file_path
    with open(file_path, "r") as f:
        lines = f.readlines()
    return list(map(lambda x: x.strip(), lines))


def main() -> None:
    file_path = "aoc_2023/day_XYZ/input/input.txt"
    lines = read_file(file_path=file_path)


if __name__ == "__main__":
    main()
