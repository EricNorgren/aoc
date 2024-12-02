import copy
from pathlib import Path


def read_file(file_path: str | Path) -> list[str]:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    assert file_path.is_file(), file_path
    with open(file_path, "r") as f:
        lines = f.readlines()
    return list(map(lambda x: x.strip(), lines))


def sign(x: int) -> int:
    if x > 0:
        return 1
    else:
        return -1


def check_valid_report(report_ints: list[int]) -> bool:
    trend = 0
    for level_first, level_second in zip(report_ints[:-1], report_ints[1:]):
        diff = level_second - level_first
        if not (1 <= abs(diff) <= 3):
            return False
        if trend == 0:
            trend = sign(diff)
        elif trend != sign(diff):
            return False
    return True

def solution_first(lines: list[str]) -> None:
    print("FIRST")
    num_valid_reports = 0
    for report in lines:
        report_ints = list(map(lambda x: int(x), report.split()))
        is_valid_report = check_valid_report(report_ints)
        if is_valid_report:
            num_valid_reports += 1
    print(f"{num_valid_reports=}")


def solution_second(lines: list[str]) -> None:
    print("SECOND")
    num_valid_reports = 0
    for report in lines:
        report_ints_original = list(map(lambda x: int(x), report.split()))
        remove_index = 0
        report_ints = report_ints_original
        report_ints_length = len(report_ints)
        is_valid_report = check_valid_report(report_ints)
        while not is_valid_report and remove_index < report_ints_length:
            report_ints = copy.deepcopy(report_ints_original)
            report_ints.pop(remove_index)
            remove_index += 1
            is_valid_report = check_valid_report(report_ints)
        if is_valid_report:
            num_valid_reports += 1
    print(f"{num_valid_reports=}")


def main() -> None:
    file_path = "input/mini_input.txt"
    file_path = "input/input.txt"

    lines = read_file(file_path=file_path)
    print(lines[0])
    solution_first(lines)
    solution_second(lines)


if __name__ == "__main__":
    main()
