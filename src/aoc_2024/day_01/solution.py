from collections import defaultdict
from pathlib import Path
import numpy as np


def read_file(file_path: str | Path) -> list[str]:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    assert file_path.is_file(), file_path
    with open(file_path, "r") as f:
        lines = f.readlines()
    return list(map(lambda x: x.strip(), lines))


def solution_first(lines: list[str]) -> None:
    print("\nFIRST")
    left = []
    right = []
    for line in lines:
        left_num, right_num = map(lambda x: int(x.strip()), line.split())
        left.append(left_num)
        right.append(right_num)
    left_arr = np.array(sorted(left))
    right_arr = np.array(sorted(right))
    dist = np.sum(np.abs(left_arr-right_arr))
    # print(left_arr)
    # print(right_arr)
    print(dist)

def solution_second(lines: list[str]) -> None:
    print("\nSECOND")
    left: dict[int, int] = defaultdict(int)
    right: dict[int, int] = defaultdict(int)
    for line in lines:
        left_num, right_num = map(lambda x: int(x.strip()), line.split())
        left[left_num] += 1
        right[right_num] += 1
    # print(left)
    # print(right)
    total = 0
    for value, num_occurances in left.items():
        if right_num_occurances := right.get(value):
            total += value * num_occurances * right_num_occurances
    print(total)


def main() -> None:
    file_path = "input/mini_input.txt"
    file_path = "input/input.txt"

    lines = read_file(file_path=file_path)
    print(lines[0])
    solution_first(lines)
    solution_second(lines)


if __name__ == "__main__":
    main()
