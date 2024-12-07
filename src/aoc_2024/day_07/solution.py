from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm


def read_file(file_path: str | Path) -> list[str]:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    assert file_path.is_file(), file_path
    with open(file_path, "r") as f:
        lines = f.readlines()
    return list(map(lambda x: x.strip(), lines))


def add(x: int, y: int) -> int:
    return x + y


def mul(x: int, y: int) -> int:
    return x * y


def concat(x: int, y: int) -> int:
    return int(f"{x}{y}")


@dataclass(frozen=True)
class EquationStates:
    current_index: int
    current_value: int


def equation_is_valid(
    input_values: list[int],
    current_index: int,
    current_value: int,
    target_value: int,
    operators: Sequence[Callable[[int, int], int]],
    verbose: bool = False,
) -> bool:
    if current_value == target_value and current_index == (len(input_values)):
        if verbose:
            print(f"FOUND MATCH current_value == target_value")
        return True
    if current_value > target_value or current_index >= len(input_values):
        return False
    current_input_value = input_values[current_index]
    if verbose:
        print(f"DEEPER:{current_index=}, {current_value=}, {current_input_value=}, {target_value=}")
    for operator in operators:
        next_current_value = operator(current_value, current_input_value)
        if verbose:
            print(
                f"{current_index=}::{operator.__name__}({current_value=}, {current_input_value=}) = {next_current_value=}"
            )
        op_is_valid = equation_is_valid(
            input_values=input_values,
            current_index=current_index + 1,
            current_value=next_current_value,
            target_value=target_value,
            operators=operators,
            verbose=verbose,
        )
        if op_is_valid:
            return True
    return False


def solution_first(lines: list[str]) -> None:
    print("FIRST")
    verbose = False
    operators = [add, mul]
    num_valid = 0
    sum_valid = 0
    for line in lines:
        target_value_str, input_values_str = line.split(": ")
        target_value = int(target_value_str)
        input_values = list(map(lambda x: int(x), input_values_str.split(" ")))
        if verbose:
            print()
            print(target_value, input_values)
        is_valid = equation_is_valid(
            input_values=input_values,
            current_index=0,
            current_value=0,
            target_value=target_value,
            operators=operators,
            verbose=verbose,
        )
        if is_valid:
            num_valid += 1
            sum_valid += target_value
    print(f"{num_valid=}, {sum_valid=}")


def solution_second(lines: list[str]) -> None:
    print("SECOND")
    verbose = False
    operators = [concat, add, mul]
    num_valid = 0
    sum_valid = 0

    for line in tqdm(lines):
        target_value_str, input_values_str = line.split(": ")
        target_value = int(target_value_str)
        input_values = list(map(lambda x: int(x), input_values_str.split(" ")))
        if verbose:
            print()
            print(target_value, input_values)
        is_valid = equation_is_valid(
            input_values=input_values,
            current_index=0,
            current_value=0,
            target_value=target_value,
            operators=operators,
            verbose=verbose,
        )

        if is_valid:
            num_valid += 1
            sum_valid += target_value
    print(f"{num_valid=}, {sum_valid=}")


def main() -> None:
    file_path = "input/mini_input.txt"
    file_path = "input/input.txt"

    lines = read_file(file_path=file_path)
    print(lines[0])
    solution_first(lines)
    solution_second(lines)


if __name__ == "__main__":
    main()
