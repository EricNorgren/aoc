from pathlib import Path
from typing import Union, List, Tuple

from scipy.signal import convolve2d
import numpy as np
import numpy.typing as npt
import pandas as pd


def read_file(file_path: Union[str | Path]) -> List[str]:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    assert file_path.is_file(), file_path
    with open(file_path, "r") as f:
        lines = f.readlines()
    return list(map(lambda x: x.strip(), lines))


def is_int(x: str) -> bool:
    return x.isdigit()


def is_symbol(x: str) -> bool:
    return not is_int(x) and not "." in x


def is_star_symbol(x: str) -> bool:
    return x == "*"


def find_number_mask(digit_mask: npt.NDArray[np.int32], adjacent_idx: int) -> List[int]:
    number_mask = set()
    current_idx = adjacent_idx
    while mask_value := digit_mask[current_idx]:
        # check backwards
        number_mask.add(current_idx)
        current_idx -= 1
        if current_idx < 0:
            break

    current_idx = adjacent_idx
    while mask_value := digit_mask[current_idx]:
        # check forwards
        number_mask.add(current_idx)
        current_idx += 1
        if current_idx >= len(digit_mask):
            break
    return sorted(number_mask)


def get_pruned_numbers_list(adjacent_mask: npt.NDArray[np.bool_]) -> List[int]:
    adjacent_indices_pruned = []
    adjacent_indices = np.argwhere(adjacent_mask)
    if len(adjacent_indices) > 0:
        # keep one adjacent idx per number
        adjacent_indices_pruned.append(adjacent_indices[0][0])
        for adjacent_idx, next_adjacent_idx in zip(adjacent_indices, adjacent_indices[1:]):
            adjacent_idx = adjacent_idx[0]
            next_adjacent_idx = next_adjacent_idx[0]
            if adjacent_idx + 1 == next_adjacent_idx:
                # consecutive idx, same number
                continue
            else:
                # new number
                adjacent_indices_pruned.append(next_adjacent_idx)
    return adjacent_indices_pruned


def parse_number_from_indices_list(number_indices: List[int], df_row: Tuple[str]) -> int:
    number = int("".join([df_row[i] for i in number_indices]))
    return number


def part_one(df: pd.DataFrame, mask_is_digit: npt.NDArray[np.bool_]) -> None:
    ####################################################################################################################
    # Part one, sum numbers adjacent to symbol that is not .
    ####################################################################################################################
    # Dummy input
    # .....
    # ../..
    # 132.4

    # mask_is_symbol_adjacent
    # FTTTFF
    # FTTTFF
    # FTTTFF

    # mask_is_digit
    # FFFFFF
    # FFFFFF
    # TTTFFT

    # mask_is_digit & mask_digit_is_symbol_adjacent
    # FFFFFF
    # FFFFFF
    # FTTFFF

    # mask_desired
    # FFFFFF
    # FFFFFF
    # TTTFFF

    mask_is_symbol = df.map(is_symbol).to_numpy().astype(int)
    expand_bool_kernel = np.ones((3, 3))
    mask_is_symbol_adjacent = convolve2d(mask_is_symbol, expand_bool_kernel, mode="same")
    mask_is_symbol_adjacent = mask_is_symbol_adjacent > 0

    mask_digit_is_symbol_adjacent = mask_is_digit & mask_is_symbol_adjacent  # digits that are adjacent to symbols
    valid_numbers = []
    for digit_mask, adjacent_mask, (_, df_row) in zip(
        mask_is_digit.astype(int), mask_digit_is_symbol_adjacent.astype(int), df.iterrows()
    ):
        adjacent_indices_pruned = get_pruned_numbers_list(
            adjacent_mask
        )  # each element in the list represents the index for a distinct word
        for adjacent_idx in adjacent_indices_pruned:
            number_indices = find_number_mask(digit_mask, adjacent_idx)
            number = parse_number_from_indices_list(number_indices, df_row)
            valid_numbers.append(number)

    print("part one:", sum(valid_numbers))  # 556367


def part_two(df: pd.DataFrame, mask_is_digit: npt.NDArray[np.bool_]) -> None:
    ####################################################################################################################
    # Part two, sum of product of exactly two numbers that are adjacent to *
    ####################################################################################################################
    mask_is_star_symbol = df.map(is_star_symbol).to_numpy().astype(int)

    # Idea: create number_instance_segementation over numbers, 0 is no number, 1 and higher represents numbers
    # Loop over np.argwhere(mask_is_star_symbol), check for number of unique neighbords in corresponding 3x3 slice from number_instance_segementation
    # if number of uniques (excluding the id 0) is equal to two, we have a "cog". Parse these two number, multiply, and add to running sum.
    # Dummy input
    # 756..
    # ..*.*
    # 132.4

    # mask_is_star_symbol
    # 00000
    # 00101
    # 00000

    # number_instance_segementation
    # 11100
    # 00000
    # 22203
    number_instance_segementation = np.zeros_like(mask_is_digit, dtype=int)
    number_counter = 1
    for row_index, (digit_mask, (_, df_row)) in enumerate(zip(mask_is_digit.astype(int), df.iterrows())):
        digit_indices_pruned = get_pruned_numbers_list(
            digit_mask
        )  # each element in the list represents the index for a distinct word
        for adjacent_idx in digit_indices_pruned:
            number_mask = find_number_mask(digit_mask, adjacent_idx)
            number_instance_segementation[row_index, number_mask] = number_counter
            number_counter += 1

    star_indices = np.argwhere(mask_is_star_symbol)
    total_sum = 0
    if len(star_indices) > 0:
        for row_index, column_index in star_indices:
            row_min, row_max = np.clip(row_index - 1, a_min=0, a_max=number_instance_segementation.shape[0]), np.clip(
                row_index + 1, a_min=0, a_max=number_instance_segementation.shape[0]
            )
            col_min, col_max = np.clip(
                column_index - 1, a_min=0, a_max=number_instance_segementation.shape[1]
            ), np.clip(column_index + 1, a_min=0, a_max=number_instance_segementation.shape[1])
            number_instance_segementation_slice = number_instance_segementation[
                row_min : row_max + 1, col_min : col_max + 1
            ]
            uniques = set(np.unique(number_instance_segementation_slice))
            uniques.remove(0)  # zero will always be present since we are looking at a slice around a * token
            if len(uniques) == 2:
                product = 1
                for number_id in uniques:
                    number_mask = number_instance_segementation == number_id
                    number_indices = np.argwhere(number_mask)
                    df_row_index = number_indices[0][0]
                    df_row = df.iloc[df_row_index, :]
                    col_indices = [idx[1] for idx in number_indices]
                    parsed_number = parse_number_from_indices_list(col_indices, df_row=df_row)
                    product *= parsed_number
                total_sum += product

    print("part two:", total_sum)  # 467835


def main() -> None:
    file_path = "aoc_2023/day_03/input/input.txt"
    # file_path = "aoc_2023/day_03/input/mini_input.txt"
    lines = read_file(file_path=file_path)

    lines_split = [[*line] for line in lines]
    df = pd.DataFrame(lines_split)
    mask_is_digit = df.map(is_int).to_numpy().astype(bool)

    part_one(df=df, mask_is_digit=mask_is_digit)
    part_two(df=df, mask_is_digit=mask_is_digit)


if __name__ == "__main__":
    main()
