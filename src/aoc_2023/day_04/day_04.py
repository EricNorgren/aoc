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


def calc_value(num_matches: int) -> int:
    if num_matches <= 0:
        return 0
    else:
        value: int = 2 ** (num_matches - 1)
        return value


def parse_card(line: str) -> int:
    # Parse the card, calculate number of matches
    card_index_str, game_str = line.split(":")
    drawn_numbers_str, winning_numbers_str = game_str.split(" | ")
    drawn_numbers_list = [x for x in drawn_numbers_str.split(" ") if len(x) > 0]
    winning_numbers_list = [x for x in winning_numbers_str.split(" ") if len(x) > 0]
    drawn_numbers_set = set(drawn_numbers_list)
    winning_numbers_set = set(winning_numbers_list)
    matching_numbers_set = drawn_numbers_set.intersection(winning_numbers_set)
    num_matches = len(matching_numbers_set)
    return num_matches


def main() -> None:
    file_path = "aoc_2023/day_04/input/input.txt"
    # file_path = "aoc_2023/day_04/input/mini_input.txt"
    lines = read_file(file_path=file_path)
    tot_value_part_one = 0
    num_copies_and_matches = np.zeros((len(lines), 2), dtype=int)  # column 0: num_copies, column 1: num_matches
    num_copies_and_matches[:, 0] = 1  # always start with one copy
    for i, line in enumerate(lines):
        num_matches = parse_card(line)
        value = calc_value(num_matches)
        tot_value_part_one += value
        num_copies_and_matches[i, 1] = num_matches

        if num_matches > 0 and i < len(num_copies_and_matches):
            min_ind = np.clip(i + 1, 0, len(num_copies_and_matches) - 1)
            max_ind = np.clip(i + num_matches + 1, 0, len(num_copies_and_matches) - 1)
            if min_ind != max_ind:
                num_copies_and_matches[min_ind:max_ind, 0] += num_copies_and_matches[i, 0]
            else:
                # Edge case: min_ind==max_ind, creating an empty slice. Increment only the last index
                num_copies_and_matches[min_ind, 0] += num_copies_and_matches[i, 0]
    tot_num_copies = np.sum(num_copies_and_matches[:, 0])
    print(f"{tot_value_part_one=}")
    print(f"{tot_num_copies=}")


if __name__ == "__main__":
    main()
