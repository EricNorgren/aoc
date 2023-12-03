from builtins import print
from pathlib import Path
from typing import Union, List, Optional, Self, Iterable

from pydantic import BaseModel, NonNegativeInt
import pandas as pd


def read_file(file_path: Union[str | Path]) -> List[str]:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    assert file_path.is_file(), file_path
    with open(file_path, "r") as f:
        lines = f.readlines()
    return list(map(lambda x: x.strip(), lines))


class CubeDraw(BaseModel):
    game_index: NonNegativeInt
    red: Optional[NonNegativeInt] = 0
    green: Optional[NonNegativeInt] = 0
    blue: Optional[NonNegativeInt] = 0

    @classmethod
    def parse_cube_str(cls, drawn_cubes_str: str, game_index: str) -> Self:
        drawn_cubes_list = drawn_cubes_str.split(", ")  # ["A red", "B green"]
        drawn_cubes_dict = {
            (cube_split := item.split(" "))[1]: cube_split[0] for item in drawn_cubes_list
        }  # {'green': 'B', 'red': 'A'}
        drawn_cubes_dict["game_index"] = game_index
        cube = cls(**drawn_cubes_dict)
        return cube


class CubeConundrumFullGame:
    def __init__(self, game_cube_str: str):
        self.game_str = game_cube_str
        self._parse_game()

    def _parse_game(self) -> None:
        game_index_str, game_str = self.game_str.split(":")  # ["Game K", "X green, Y red, Z blue; A red, B green"]
        self.game_index_str = int(game_index_str.strip().split(" ")[-1])  # K
        self.drawn_cubes_str_list = list(
            map(lambda x: x.strip(), game_str.split("; "))
        )  # ["X green, Y red, Z blue", "A red, B green"]
        self.drawn_cubes_as_list_of_dict = []
        for drawn_cubes_str in self.drawn_cubes_str_list:
            cube = CubeDraw.parse_cube_str(drawn_cubes_str, game_index=self.game_index_str)
            self.drawn_cubes_as_list_of_dict.append(dict(cube))


def main() -> None:
    file_path = "aoc_2023/day_02/input/input.txt"
    lines = read_file(file_path=file_path)

    cube_games = []
    for line in lines:
        cube_game = CubeConundrumFullGame(line)
        cube_games.extend(cube_game.drawn_cubes_as_list_of_dict)
    df = pd.DataFrame([cube_game for cube_game in cube_games])  # dataframe containing all draws over all games
    ####################################################################################################################
    # Part one, sum of game indicies, where the game is possible given BAG_CUBE_CONTENTS
    ####################################################################################################################
    BAG_CUBE_CONTENTS = {
        "red": 12,
        "green": 13,
        "blue": 14,
    }
    print(BAG_CUBE_CONTENTS)
    max_draw_per_game_df = df.groupby("game_index").agg("max")
    masks = {}
    for color, max_num_cubes in BAG_CUBE_CONTENTS.items():
        masks[color] = max_draw_per_game_df[color] <= max_num_cubes
    masks = pd.DataFrame.from_dict(masks)
    masks["possible"] = masks.agg(lambda x: sum(x) == len(x), axis="columns")  # "and"-operator over columns...
    masks = masks.reset_index()

    print("game_index sum:", masks.loc[masks["possible"], "game_index"].sum())  # 2101

    ####################################################################################################################
    # Part two, fewest number of colors per game, calculate power over each game, return sum over all powers.
    ####################################################################################################################
    def power(x_iterable: Iterable[int]) -> int:
        power_res = 1
        for x in x_iterable:
            power_res *= x
        return power_res

    max_draw_per_game_df["power"] = max_draw_per_game_df.agg(power, axis="columns")
    print("power sum:", max_draw_per_game_df["power"].sum())  # 58269


if __name__ == "__main__":
    main()
