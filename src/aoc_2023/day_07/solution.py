import copy
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path
from typing import Union, List, Type, Dict
from itertools import product


def read_file(file_path: Union[str | Path]) -> List[str]:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    assert file_path.is_file(), file_path
    with open(file_path, "r") as f:
        lines = f.readlines()
    return list(map(lambda x: x.strip(), lines))


class GAME_RULES(Enum):
    PART_ONE = auto()
    PART_TWO = auto()


CARD_VALUES_PART_ONE = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
CARD_VALUES_PART_TWO = ["A", "K", "Q", "T", "9", "8", "7", "6", "5", "4", "3", "2", "J"]  # joker last


CARD_VALUES_PART_ONE_TO_INT = {
    card_type: int(str(value)) for value, card_type in enumerate(reversed(CARD_VALUES_PART_ONE))
}
CARD_VALUES_PART_TWO_TO_INT = {
    card_type: int(str(value)) for value, card_type in enumerate(reversed(CARD_VALUES_PART_TWO))
}


def parse_hand(hand_str: str, card_mapping: Dict[str, int]) -> List[int]:
    hand = [card_mapping[card_type] for card_type in hand_str]
    return hand


class HAND_TYPE(Enum):
    HIGH_CARD = auto()
    ONE_PAIR = auto()
    TWO_PAIR = auto()
    THREE_OF_A_KIND = auto()
    FULL_HOUSE = auto()
    FOUR_OF_A_KIND = auto()
    FIVE_OF_A_KIND = auto()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, HAND_TYPE):
            return NotImplemented
        return self.value < other.value

    @classmethod
    def from_hand_part_one(cls: Type["HAND_TYPE"], hand: List[int]) -> "HAND_TYPE":
        values_to_positions = defaultdict(list)
        for index, value in enumerate(hand):
            values_to_positions[value].append(index)
        unique_cards = [len(v) for k, v in values_to_positions.items()]

        _unique_cards = copy.deepcopy(unique_cards)
        if 5 in _unique_cards:
            return cls.FIVE_OF_A_KIND
        elif 4 in _unique_cards:
            return cls.FOUR_OF_A_KIND
        elif 3 in _unique_cards:
            if 2 in _unique_cards:
                return cls.FULL_HOUSE
            else:
                return cls.THREE_OF_A_KIND
        elif 2 in _unique_cards:
            _unique_cards.remove(2)
            if 2 in _unique_cards:
                return cls.TWO_PAIR
            else:
                return cls.ONE_PAIR
        else:
            return cls.HIGH_CARD

    @classmethod
    def from_hand_part_two(cls: Type["HAND_TYPE"], hand: List[int]) -> "HAND_TYPE":
        _joker_value = CARD_VALUES_PART_TWO_TO_INT["J"]
        if _joker_value in hand:
            # Jokers are powerful
            values_to_positions = defaultdict(list)
            for index, value in enumerate(hand):
                values_to_positions[value].append(index)
            joker_positions = values_to_positions[_joker_value]

            # loop over all jokers swapped to every other value
            num_jokers = len(values_to_positions[_joker_value])
            # possible combinations = num_jokers^num_unique_values
            exploded_jokers = product(range(len(set(hand))), repeat=num_jokers)
            # example iteration over exploded_jokers:
            # hand=[9, 4, 4, 0, 4]
            # joker_replacement_targets=(0,)
            # joker_replacement_targets=(1,)
            # joker_replacement_targets=(2,)
            # hand=[11, 9, 0, 0, 9]
            # joker_replacement_targets=(0, 0)
            # joker_replacement_targets=(0, 1)
            # joker_replacement_targets=(0, 2)
            # joker_replacement_targets=(1, 0)
            # joker_replacement_targets=(1, 1)
            # joker_replacement_targets=(1, 2)
            # joker_replacement_targets=(2, 0)
            # joker_replacement_targets=(2, 1)
            # joker_replacement_targets=(2, 2)

            target_unique_values = {
                index: value for index, value in enumerate(set(hand))
            }  # mapping index to the unique contents in hand
            # Loop over combinations of hands, keep the highest observed. BRUTE FORCE!!!
            max_seen_rank = cls.from_hand_part_one(hand)  # baseline to compare against
            for joker_replacement_targets in exploded_jokers:
                # joker_replacement_targets will be a tuple of length num_jokers.
                # each index represents which value the i:th joker should take, via target_unique_values
                temp_hand = copy.deepcopy(hand)
                for joker_write_index, joker_replacement_target in enumerate(joker_replacement_targets):
                    target_value = target_unique_values[joker_replacement_target]
                    joker_index = joker_positions[joker_write_index]
                    temp_hand[joker_index] = target_value
                temp_hand_type = cls.from_hand_part_one(temp_hand)
                if temp_hand_type > max_seen_rank:
                    max_seen_rank = temp_hand_type
            return max_seen_rank
        else:
            # No joker, proceed as usual
            return cls.from_hand_part_one(hand)


class Hand:
    def __init__(self, line: str, game_rules: GAME_RULES):
        self.line = line
        hand_str, bet_str = line.split()
        self.bet = int(bet_str)

        if game_rules == GAME_RULES.PART_ONE:
            self.line_as_ints = parse_hand(hand_str, card_mapping=CARD_VALUES_PART_ONE_TO_INT)
            self.hand_type = HAND_TYPE.from_hand_part_one(self.line_as_ints)
        if game_rules == GAME_RULES.PART_TWO:
            self.line_as_ints = parse_hand(hand_str, card_mapping=CARD_VALUES_PART_TWO_TO_INT)
            self.hand_type = HAND_TYPE.from_hand_part_two(self.line_as_ints)

    def __str__(self) -> str:
        return f"{self.line}, {self.line_as_ints}, {self.hand_type}, {self.bet}"

    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Hand):
            return NotImplemented
        if self.hand_type == other.hand_type:
            for card_value_self, card_value_other in zip(self.line_as_ints, other.line_as_ints):
                if card_value_self == card_value_other:
                    continue
                else:
                    return card_value_self < card_value_other

            return False  # They are equal
        else:
            return self.hand_type < other.hand_type

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hand):
            return NotImplemented
        return self.hand_type == other.hand_type and self.line_as_ints == other.line_as_ints


def calc_winnings(lines: List[str], game_type: GAME_RULES) -> int:
    hands = []
    for line in lines:
        hand = Hand(line, game_rules=game_type)
        hands.append(hand)
    sorted_hands = list(sorted(hands))
    sum = 0
    for index, h1 in enumerate(sorted_hands):
        winnings = (index + 1) * h1.bet
        sum += winnings
    return sum


def main() -> None:
    file_path = "aoc_2023/day_07/input/input.txt"
    # file_path = "aoc_2023/day_07/input/mini_input.txt"
    lines = read_file(file_path=file_path)
    winnings_p1 = calc_winnings(lines, game_type=GAME_RULES.PART_ONE)
    print(f"{winnings_p1=}") # 247815719
    winnings_p2 = calc_winnings(lines, game_type=GAME_RULES.PART_TWO)
    print(f"{winnings_p2=}") # 248747492


if __name__ == "__main__":
    main()
