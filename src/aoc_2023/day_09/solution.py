from enum import Enum, auto
from pathlib import Path
from typing import Union, List

import numpy as np


def read_file(file_path: Union[str | Path]) -> List[str]:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    assert file_path.is_file(), file_path
    with open(file_path, "r") as f:
        lines = f.readlines()
    return list(map(lambda x: x.strip(), lines))


class DIRECTION(Enum):
    FORWARD = auto()
    BACKWARD = auto()


def predict_next_number(line: str, direction: DIRECTION) -> int:
    number_seqeuence = np.array(line.split(), dtype=np.int_)
    current_sequence = number_seqeuence
    sequences = []
    while not (current_sequence == 0).all():
        sequences.append(current_sequence)
        current_sequence = current_sequence[1:] - current_sequence[:-1]

    if direction == DIRECTION.FORWARD:
        next_number = sum(s[-1] for s in sequences)
    elif direction == DIRECTION.BACKWARD:
        sequences = list(reversed(sequences))
        prev_deriv = 0
        next_numbers = [prev_deriv := deriv_lower[0] - prev_deriv for deriv_lower in sequences]
        next_number = next_numbers[-1]
    return next_number


def main() -> None:
    file_path = "aoc_2023/day_09/input/mini_input.txt"
    file_path = "aoc_2023/day_09/input/input.txt"
    lines = read_file(file_path=file_path)

    print(lines[0])
    tot_sum_next = 0
    tot_sum_prev = 0
    for line in lines:
        next_number = predict_next_number(line, direction=DIRECTION.FORWARD)
        prev_number = predict_next_number(line, direction=DIRECTION.BACKWARD)
        tot_sum_next += next_number
        tot_sum_prev += prev_number
    print(f"{tot_sum_next=}") # 1887980197
    print(f"{tot_sum_prev=}") # 990


if __name__ == "__main__":
    main()
