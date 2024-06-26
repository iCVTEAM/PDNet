from enum import Enum, unique


@unique
class DatasetType(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2
