import copy
import math
from collections import defaultdict
from pathlib import Path
from typing import Union, List, Dict, Set, Type
from itertools import cycle

import numpy as np
import numpy.typing as npt

from numba import njit

def read_file(file_path: Union[str | Path]) -> List[str]:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    assert file_path.is_file(), file_path
    with open(file_path, "r") as f:
        lines = f.readlines()
    return list(map(lambda x: x.strip(), lines))


class Node:
    def __init__(self, line: str):
        self.line = line

        self.value, children_str = line.split(" = ")
        self.left_str, self.right_str = children_str[1:-1].split(", ")

    def __str__(self) -> str:
        return f"{self.value}, {self.left_str}, {self.right_str}"

    @property
    def L(self) -> str:
        return self.left_str

    @property
    def R(self) -> str:
        return self.right_str

def part_one(instructions: str, nodes: Dict[str, Node]) -> None:
    num_steps = 0 
    current_node_str = "AAA"
    current_node = nodes[current_node_str]

    instruction_iterator = cycle(instructions)  # Infinite repeats
    while current_node_str != "ZZZ":
        instruction = next(instruction_iterator)
        num_steps += 1;
        current_node_str = getattr(current_node, instruction)
        current_node = nodes[current_node_str]

    print(f"part one {num_steps=}")# 19099

def is_terminal_state(current_nodes: List[str]) -> bool:
    return sum([current_node[-1] == "Z" for current_node in current_nodes]) == len(current_nodes)



def part_two_naive(instructions: str, nodes: Dict[str, Node]) -> None:
    # 10M in 20 seconds, stopped at 645 000 000
    # 500000/second
    num_steps = 0
    current_node_strs = [current_node for current_node in nodes.keys() if current_node[-1] == "A"] # starting node
    current_nodes = [nodes[current_node_str] for current_node_str in current_node_strs]


    instruction_iterator: Type[cycle] = cycle(instructions)  # Infinite repeats
    while not is_terminal_state(current_node_strs):
        instruction = next(instruction_iterator)
        num_steps += 1

        current_node_strs = [getattr(current_node, instruction) for current_node in current_nodes]
        current_nodes = [nodes[current_node_str] for current_node_str in current_node_strs]
        if num_steps % 1e6 == 0:
            print("running... ", num_steps)

    print("PART TWO DONE ", num_steps)

@njit
def part_two_naive_numba(instructions: str, network_map: List[str]) -> None:
    nodes = {}
    for line in network_map:
        value, children_str = line.split(" = ")
        left_str, right_str = children_str[1:-1].split(", ")
        nodes[value] = (left_str, right_str)

    instruction_to_int_mapping = {"L": 0, "R": 1}

    current_node_strs = [current_node for current_node in nodes.keys() if current_node[-1] == "A"] # starting node
    current_nodes = [nodes[current_node_str] for current_node_str in current_node_strs]

    num_steps = 0
    instruction_index = 0
    done = sum([current_node[-1] == "Z" for current_node in current_nodes]) == len(current_nodes)
    while not done:
        instruction = instructions[instruction_index]
        instruction_in_tuple =  instruction_to_int_mapping[instruction]
        num_steps += 1
        instruction_index += 1
        if instruction_index >= len(instructions):
            instruction_index = 0

        current_node_strs = [current_node[instruction_in_tuple] for current_node in current_nodes]
        current_nodes = [nodes[current_node_str] for current_node_str in current_node_strs]
        done = sum([current_node[-1] == "Z" for current_node in current_nodes]) == len(current_nodes)
        if num_steps % 1e6 == 0:
            print("running... ", num_steps)

    print("PART TWO DONE ", num_steps)


def find_terminal_node_strs(current_node_strs: List[str]) -> List[str]:
    terminal_strs = [current_node_str for current_node_str in current_node_strs if current_node_str[-1]=="Z"]
    return terminal_strs

def part_two_lcm(instructions: str, nodes: Dict[str, Node]) -> None:
    num_steps = 0
    current_node_strs = [current_node for current_node in nodes.keys() if current_node[-1] == "A"] # starting node
    current_nodes = [nodes[current_node_str] for current_node_str in current_node_strs]

    first_solution_at_step: Dict[str, int] = {}
    num_starting_nodes = len(current_node_strs)
    num_prints = 0

    instruction_iterator = cycle(instructions)  # Infinite repeats
    while len(first_solution_at_step) != num_starting_nodes:
        terminal_node_strs = find_terminal_node_strs(current_node_strs)
        for terminal_node_str in terminal_node_strs:
            if not terminal_node_str in first_solution_at_step.keys():
                first_solution_at_step[terminal_node_str] = num_steps
                current_node_strs.remove(terminal_node_str)
                current_nodes.remove(nodes[terminal_node_str])
        instruction = next(instruction_iterator)
        num_steps += 1
        current_node_strs = [getattr(current_node, instruction) for current_node in current_nodes]
        current_nodes = [nodes[current_node_str] for current_node_str in current_node_strs]

    print(f"{first_solution_at_step=}")
    print(f"{first_solution_at_step.values()=}")
    answer = math.lcm(*list(first_solution_at_step.values()))
    print(f"part two {answer=}") # 17099847107071

def main() -> None:
    file_path = "aoc_2023/day_08/input/mini_input_2.txt"
    file_path = "aoc_2023/day_08/input/mini_input.txt"
    file_path = "aoc_2023/day_08/input/part_2_mini_input.txt"
    file_path = "aoc_2023/day_08/input/input.txt"
    lines = read_file(file_path=file_path)
    print(lines[0])
    instructions = lines[0]
    network_map = lines[2:]
    nodes = {}
    for line in network_map:
        node = Node(line)
        nodes[node.value] = node
        print(node)

    print()
    part_one(instructions=instructions, nodes=nodes)
    print()
    # part_two_naive(instructions=instructions, nodes=nodes) # requires about a year to run
    # part_two_naive_numba(instructions=instructions, network_map=network_map)
    part_two_lcm(instructions=instructions, nodes=nodes)


if __name__ == "__main__":
    main()
