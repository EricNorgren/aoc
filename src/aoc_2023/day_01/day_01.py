import copy
from pathlib import Path
from typing import List, Union, Optional

import re
from re import Pattern, Match


def read_file(file_path: Union[str | Path]) -> List[str]:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    assert file_path.is_file(), file_path
    with open(file_path, "r") as f:
        lines = f.readlines()
    return list(map(lambda x: x.strip(), lines))


def find_index(in_str: str, pattern: Pattern[str]) -> Match[str]:
    match: Match[str] | None = pattern.search(in_str)
    assert match is not None, match
    return match

def match_to_int(match: Match[str]) -> str:
    str_to_int = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }
    str_to_int_with_backwards = copy.deepcopy(str_to_int)
    # Add reverse matching
    for str_digit, value in str_to_int.items():
        str_to_int_with_backwards[str_digit[::-1]] = value

    match_str = match[0]
    if match_str in str_to_int_with_backwards.keys():
        return str_to_int_with_backwards[match_str]
    else:
        return match_str

def decipher_calibration_document(lines: List[str], match_regex: str, match_regex_backward: Optional[str] = None, verbose: Optional[bool] = False) -> int:
    regex_pattern_forward: Pattern[str] = re.compile(pattern=match_regex)

    if match_regex_backward is None:
        match_regex_backward = match_regex
    regex_pattern_backward: Pattern[str] = re.compile(pattern=match_regex_backward)

    running_sum = 0
    for line in lines:
        match_start = find_index(line, regex_pattern_forward)
        match_end = find_index(line[::-1], regex_pattern_backward)  # line[::-1] reverses the string
        number_start = match_to_int(match_start)
        number_end = match_to_int(match_end)
        number_match = int(f"{number_start}{number_end}")
        running_sum += number_match
        if verbose:
            print(f"{number_start} {number_end} {number_match}, {line=}")
    return running_sum

def main() -> None:
    file_path = "aoc_2023/day_01/input/input.txt"
    lines = read_file(file_path=file_path)
    ####################################################################################################################
    # Part one, find digits only
    ####################################################################################################################
    match_regex_part_one = "[0-9]"
    # 54338
    print(decipher_calibration_document(lines=lines, match_regex=match_regex_part_one))

    ####################################################################################################################
    # Part two, find digits and digits as words
    ####################################################################################################################
    # Build backwards matching strategy ([0-9]|one|two|...)
    digits_as_str_list = "one|two|three|four|five|six|seven|eight|nine".split("|")
    match_regex_part_two_forward_no_parenthesis = "|".join([match_regex_part_one, *digits_as_str_list])
    match_regex_part_two_forward = f"({match_regex_part_two_forward_no_parenthesis})"

    # Build backwards matching strategy ([0-9]|eno|owt|...)
    match_regex_part_two_backward_list = [match_regex_part_one]
    for digits_as_str in digits_as_str_list:
        match_regex_part_two_backward_list.append(digits_as_str[::-1]) # reverse string: one --> eno, two --> owt, ...
    match_regex_part_two_forward_no_parenthesis = "|".join(match_regex_part_two_backward_list)
    match_regex_part_two_backward = f"({match_regex_part_two_forward_no_parenthesis})"

    # 53389
    print(decipher_calibration_document(lines=lines, match_regex=match_regex_part_two_forward, match_regex_backward=match_regex_part_two_backward))


if __name__ == "__main__":
    main()
