from enum import Enum

class Action(Enum):
    BUY = 1
    FLAT = 0
    SELL = -1

def get_action_code(a: Action)->int:
    return {
        Action.FLAT: 0,
        Action.BUY: 1,
        Action.SELL: 2
    }[a]

def convert_to_action(a: int) -> Action:
    return {
        0: Action.FLAT,
        1: Action.BUY,
        2: Action.SELL
    }[a]