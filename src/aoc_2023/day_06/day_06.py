from builtins import set
from pathlib import Path
from typing import Union, List, Tuple, TypeVar

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.typing as npt


def read_file(file_path: Union[str | Path]) -> List[str]:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    assert file_path.is_file(), file_path
    with open(file_path, "r") as f:
        lines = f.readlines()
    return list(map(lambda x: x.strip(), lines))

T = TypeVar("T", int, npt.NDArray[np.int32], npt.NDArray[np.longlong])

def calc_distance(seconds_held: T, race_seconds_length: T) -> T:

    # assert (seconds_held >= 0).all(), seconds_held
    # assert (race_seconds_length >= 0).all(), race_seconds_length
    # assert (race_seconds_length >= seconds_held).all(), f"{seconds_held=}, {race_seconds_length=}"

    t_0 = seconds_held
    v_0 = 0
    t_1 = race_seconds_length - seconds_held
    v_1 = seconds_held
    return t_0 * v_0 + t_1 * v_1

def create_plot(rows: int = 1, cols: int = 1, width: int = 15, height: int = 7) -> Union[Tuple[plt.Figure, List[plt.Axes]]]:
    fig, axs = plt.subplots(rows, cols, figsize=(width, height))
    if type(axs) == plt.Axes:
        axs = [axs]
    return fig, axs

def plot_all_distances(times: List[int], record_distances: List[int]):
    num_plots = len(times)
    fig, axs = create_plot(rows=num_plots, height=12)

    for time, record_distance, ax in zip(times, record_distances, axs):
        times_exploded = np.arange(time + 1, dtype=int)
        distances_exploded = []
        for t in times_exploded:
            distance = calc_distance(seconds_held=t, race_seconds_length=time)
            distances_exploded.append(distance)

        ax.grid()
        ax.axhline(c="r", y=record_distance)
        ax.scatter(times_exploded, distances_exploded)
        ax.set_xlabel("Release time (ms)")
        ax.set_ylabel("Distance (mm)")

    plt.show()

def calc_num_winning_instances(df: pd.DataFrame, time_dtype: Union[np.dtype[np.int_], np.dtype[np.longlong]]) -> pd.DataFrame:
    num_winning_distances = []
    for idx, df_row in df.iterrows():
        race_seconds_length, record_distances = df_row["race_seconds_length"], df_row["record_distances"]
        times = np.arange(race_seconds_length + 1, dtype=time_dtype)
        distances = calc_distance(seconds_held=times, race_seconds_length=race_seconds_length)
        winning_distances = distances > record_distances
        num_winning_distances.append(sum(winning_distances))
    df["num_winning_distances"] = num_winning_distances
    return df

def part_one(lines: List[str]):
    times_input = list(map(int, lines[0].split()[1:]))
    distances_input = list(map(int, lines[1].split()[1:]))
    df = pd.DataFrame.from_dict({"race_seconds_length": times_input, "record_distances": distances_input})
    df = calc_num_winning_instances(df, time_dtype=np.dtype(np.int_))  # type: ignore
    print(df)
    # plot_all_distances(times_input, distances_input)
    print("part_one:: product", np.prod(df["num_winning_distances"]))

def part_two(lines: List[str]):
    times_input = [int("".join(list(lines[0].split()[1:])))]
    distances_input = [int("".join(list(lines[1].split()[1:])))]
    df = pd.DataFrame.from_dict({"race_seconds_length": times_input, "record_distances": distances_input})
    df = calc_num_winning_instances(df, time_dtype=np.dtype(np.longlong))

    print(df)
    # plot_all_distances(times_input, distances_input)
    print("part_two:: product", np.prod(df["num_winning_distances"]))


def main() -> None:
    file_path = "aoc_2023/day_06/input/input.txt"
    # file_path = "aoc_2023/day_06/input/mini_input.txt"
    lines = read_file(file_path=file_path)
    part_one(lines)
    part_two(lines)




if __name__ == "__main__":
    main()
