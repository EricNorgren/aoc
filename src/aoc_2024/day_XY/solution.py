from pathlib import Path


def read_file(file_path: str | Path) -> list[str]:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    assert file_path.is_file(), file_path
    with open(file_path, "r") as f:
        lines = f.readlines()
    return list(map(lambda x: x.strip(), lines))


def solution_first(lines: list[str]) -> None:
    print("FIRST")

def solution_second(lines: list[str]) -> None:
    print("SECOND")


def main() -> None:
    file_path = "input/input.txt"
    file_path = "input/mini_input.txt"

    lines = read_file(file_path=file_path)
    print(lines[0])
    solution_first(lines)
    solution_second(lines)


if __name__ == "__main__":
    main()
