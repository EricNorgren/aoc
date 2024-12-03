from pathlib import Path

import re


def read_file(file_path: str | Path) -> list[str]:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    assert file_path.is_file(), file_path
    with open(file_path, "r") as f:
        lines = f.readlines()
    return list(map(lambda x: x.strip(), lines))


def parse_maybe_mul_x_y(maybe_mul_x_y: str) -> int:
    digit_regex = r"\d{1,3},\d{1,3}"
    mul_regex = r"mul\(" + digit_regex + r"\)"
    mul_matches = re.findall(mul_regex, maybe_mul_x_y)
    total = 0
    for mul in mul_matches:
        first, second = map(lambda x: int(x), re.findall(digit_regex, mul)[0].split(","))
        total += first * second
    return total


def solution_first(lines: list[str]) -> None:
    print("FIRST")
    total = 0
    for maybe_mul_x_y in lines:
        total += parse_maybe_mul_x_y(maybe_mul_x_y=maybe_mul_x_y)
    print(f"{total=}")


def solution_second(lines: list[str]) -> None:
    print("SECOND")
    dont_regex = r"don\'t\(\)"
    do_regex = r"do\(\)"
    to_delete_regex = dont_regex + r".*?" + do_regex

    lines_joined = "".join(lines)
    total = 0
    for line in [lines_joined]:
        delete_matches_iter = re.finditer(to_delete_regex, line)
        prev_index = 0
        end = 0
        keep_dos = []
        for delete_match in delete_matches_iter:
            start = delete_match.start()
            end = delete_match.end()
            keep_dos.append(line[prev_index:start])
            prev_index = end
        tail = line[end:].split("don't()", maxsplit=1)[0]
        keep_dos.append(tail)
        for maybe_mul_x_y in keep_dos:
            total += parse_maybe_mul_x_y(maybe_mul_x_y=maybe_mul_x_y)
    print(f"{total=}")


def main() -> None:
    file_path = "input/mini_input.txt"
    file_path = "input/input.txt"

    lines = read_file(file_path=file_path)
    # print(lines[0])
    solution_first(lines)  # 187825547
    solution_second(lines)  # 85508223


if __name__ == "__main__":
    main()
