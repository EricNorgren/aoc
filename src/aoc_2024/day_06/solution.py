import copy
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from tqdm import tqdm
import numpy as np
import numpy.typing as npt


def read_file(file_path: str | Path) -> list[str]:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    assert file_path.is_file(), file_path
    with open(file_path, "r") as f:
        lines = f.readlines()
    return list(map(lambda x: x.strip(), lines))


# [Y, X], positive down, positive right
UP_VECTOR = np.array([-1, 0], dtype=np.int_)
RIGHT_VECTOR = np.array([0, 1], dtype=np.int_)
DOWN_VECTOR = np.array([1, 0], dtype=np.int_)
LEFT_VECTOR = np.array([0, -1], dtype=np.int_)
UP_ARROW = "^"
RIGHT_ARROW = ">"
DOWN_ARROW = "v"
LEFT_ARROW = "<"
ARROW_DIRECTION_ORDERED = [
    (UP_ARROW, UP_VECTOR),
    (RIGHT_ARROW, RIGHT_VECTOR),
    (DOWN_ARROW, DOWN_VECTOR),
    (LEFT_ARROW, LEFT_VECTOR),
]
DIRECTIONS_ORDERED = [x[1] for x in ARROW_DIRECTION_ORDERED]
ARROWS_ORDERED = [x[0] for x in ARROW_DIRECTION_ORDERED]

class Direction:
    def __init__(self, initial_arrow: str = "^") -> None:
        self.current_index = ARROWS_ORDERED.index(initial_arrow)

    @property
    def current_direction_vector(self) -> npt.NDArray[np.int_]:
        return DIRECTIONS_ORDERED[self.current_index]

    @property
    def current_arrow(self) -> str:
        return ARROWS_ORDERED[self.current_index]

    def rotate(self) -> None:
        self.current_index = (self.current_index + 1) % len(DIRECTIONS_ORDERED)

    def __str__(self) -> str:
        direction = self.current_direction_vector
        arrow = self.current_arrow
        return f"Direction({self.current_index}, {direction=}, {arrow=})"


class EndPositionType(Enum):
    PILLAR = auto()
    OUTSIDE = auto()


@dataclass
class EndPositionResult:
    end_position: npt.NDArray[np.int_]
    end_position_type: EndPositionType


def get_end_index(
    pillars: npt.NDArray[np.int_],
    guard_position: npt.NDArray[np.int_],
    current_direction: Direction,
    board_size: tuple[int, int],
) -> EndPositionResult:
    current_direction_vector = current_direction.current_direction_vector
    changing_axis = np.transpose(np.nonzero(current_direction_vector)).squeeze()
    keeping_axis = np.transpose(np.nonzero(current_direction_vector == 0)).squeeze()

    pillar_candidates_mask = (guard_position[keeping_axis] == pillars[:, keeping_axis]).squeeze()
    pillar_candidates_idx = np.argwhere(pillar_candidates_mask).squeeze(1)
    pillar_candidates = pillars[pillar_candidates_idx]

    closest_distance = np.inf
    end_index = np.zeros(2, dtype=np.int_) - 1  # Dummy placeholder until used
    end_index_changing_axis = -1

    for pillar_candidate in pillar_candidates:
        pillar_vector = pillar_candidate - guard_position
        new_distance = (
            np.abs(pillar_vector[changing_axis]) - 1
        )  # subtracting one since we want to stop just before pillar

        direction_sign = np.sign(current_direction.current_direction_vector[changing_axis])
        pillar_sign = np.sign(pillar_vector[changing_axis])
        # print(f"{direction_sign=}, {pillar_sign=}, {current_direction.current_arrow}")
        # print(f"{pillar_candidate=}, {new_distance=}")
        if new_distance < closest_distance and direction_sign == pillar_sign:
            closest_distance = new_distance
            end_index_changing_axis = (
                guard_position[changing_axis]
                + current_direction.current_direction_vector[changing_axis] * closest_distance
            )
            # print(f"FOUND NEW PILLAR: {pillar_candidate=}, {new_distance=}")
    end_index[keeping_axis] = guard_position[keeping_axis]
    if end_index_changing_axis < 0:
        end_position_type = EndPositionType.OUTSIDE
        edge_position = np.clip(
            guard_position + current_direction.current_direction_vector * max(board_size) * 2,
            a_min=(0, 0),
            a_max=np.array(board_size, dtype=np.int_) - 1,
        )
        end_index_changing_axis = edge_position[changing_axis]
    else:
        end_position_type = EndPositionType.PILLAR
    end_index[changing_axis] = end_index_changing_axis
    assert all([x >= 0 for x in end_index]), end_index
    return EndPositionResult(end_position=end_index, end_position_type=end_position_type)


class GameBoard:
    def __init__(self, lines: list[str]) -> None:
        self.board = np.array(list(map(lambda x: list(map(lambda y: y.split(), x)), lines))).squeeze()
        self.board_traveled = np.zeros_like(self.board)
        self.board_traveled[:, :] = "."
        initial_arrow = None
        for arrow in ARROWS_ORDERED:
            if arrow in self.board:
                initial_arrow = arrow
        if initial_arrow is None:
            raise ValueError(f"Invalid initial board: {str(self)}")
        self.current_direction = Direction(initial_arrow=initial_arrow)
        guard_position = np.transpose(np.nonzero(self.board == self.current_direction.current_arrow))
        assert len(guard_position) == 1
        self.start_guard_position = guard_position[0]
        self.guard_position = guard_position[0]
        self.pillars = np.transpose(np.nonzero(self.board == "#"))

    @property
    def board_size(self) -> tuple[int, int]:
        board_shape: tuple[int, int] = self.board.shape  # type: ignore
        return board_shape

    def advance_state(self) -> EndPositionResult:
        # print("ADVANCING")
        end_position_result = get_end_index(
            pillars=self.pillars,
            guard_position=self.guard_position,
            current_direction=self.current_direction,
            board_size=self.board.shape,
        )
        self.update_board(start_position=self.guard_position, end_position_result=end_position_result)
        return end_position_result

    def update_board(self, start_position: npt.NDArray[np.int_], end_position_result: EndPositionResult) -> None:
        changing_axis = np.transpose(np.nonzero(self.current_direction.current_direction_vector)).squeeze()
        keeping_axis = np.transpose(np.nonzero(self.current_direction.current_direction_vector == 0)).squeeze()
        end_position = end_position_result.end_position
        start_idx = min(start_position[changing_axis], end_position[changing_axis])
        end_idx = max(start_position[changing_axis], end_position[changing_axis]) + 1
        changing_slice = slice(start_idx, end_idx)
        if changing_axis == 0:
            slice_y = changing_slice
            slice_x = start_position[keeping_axis]
        else:
            slice_y = start_position[keeping_axis]
            slice_x = changing_slice

        self.board[slice_y, slice_x] = "X"
        self.board_traveled[slice_y, slice_x] = "X"
        self.guard_position = end_position
        self.current_direction.rotate()
        if end_position_result.end_position_type == EndPositionType.PILLAR:
            self.board[end_position[0], end_position[1]] = self.current_direction.current_arrow

    def solution_first(self, verbose=False) -> int:
        while True:
            if verbose:
                print(f"calc::{self.guard_position=}")
            end_position = self.advance_state()
            if verbose:
                print(f"calc::{end_position=}")
                print(f"calc::board:\n{self.board}")
                print(f"calc::current_direction:{self.current_direction}")
                print()
            if end_position.end_position_type == EndPositionType.OUTSIDE:
                num_x: int = np.sum(self.board == "X")
                return num_x

    def __str__(self) -> str:
        return f"GameBoard(\n{self.board}\n{self.current_direction}\n{self.guard_position=}\n{self.pillars=})"

@dataclass(frozen=True)
class GuardState:
    position_0: int
    position_1: int
    direction: str

class TimeTravel:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        gameboard = GameBoard(lines=lines)
        gameboard.solution_first(verbose=False)
        candidate_obstacle_positions_x = gameboard.board_traveled
        candidate_obstacle_positions_x[gameboard.start_guard_position[0], gameboard.start_guard_position[1]] = "."
        self.candidate_obstacle_positions = np.transpose(np.nonzero(candidate_obstacle_positions_x == "X"))

    def solution_second(self) -> int:
        obstacle_positions = []
        for candidate_obstacle_position in tqdm(self.candidate_obstacle_positions):
            lines_with_obstacle = copy.deepcopy(self.lines)
            line_y = lines_with_obstacle[candidate_obstacle_position[0]]
            lines_with_obstacle[candidate_obstacle_position[0]] = line_y[:candidate_obstacle_position[1]] + "#" + line_y[candidate_obstacle_position[1] + 1:]
            gameboard = GameBoard(lines=lines_with_obstacle)
            guard_state = GuardState(position_0=gameboard.guard_position[0], position_1=gameboard.guard_position[1],
                                     direction=gameboard.current_direction.current_arrow)
            visited_states = set()
            visited_states.add(guard_state)
            while True:
                end_position = gameboard.advance_state()
                guard_state = GuardState(position_0=gameboard.guard_position[0], position_1=gameboard.guard_position[1],
                                         direction=gameboard.current_direction.current_arrow)
                if guard_state in visited_states:
                    # found circle
                    obstacle_positions.append(candidate_obstacle_position)
                    # print(f"FOUND OBSTACLE: {candidate_obstacle_position}")
                    break
                else:
                    visited_states.add(guard_state)
                if end_position.end_position_type == EndPositionType.OUTSIDE:
                    break
        # print(f"{obstacle_positions=}")
        return len(obstacle_positions)


def solution_first(lines: list[str]) -> None:
    print("FIRST")
    gameboard = GameBoard(lines=lines)
    num_x = gameboard.solution_first(verbose=False)
    print(f"{num_x=}")


def solution_second(lines: list[str]) -> None:
    print("SECOND")
    timetravel = TimeTravel(lines=lines)
    num_obstacles = timetravel.solution_second()
    print(f"{num_obstacles=}")


def main() -> None:
    file_path = "input/mini_input.txt"
    file_path = "input/input.txt"

    lines = read_file(file_path=file_path)
    print(lines[0])
    solution_first(lines)  # 5305
    # Second, brute force. Takes about 90 seconds.
    solution_second(lines) # 2143


if __name__ == "__main__":
    main()
