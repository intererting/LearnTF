def test(a: int, b: float, c: str,
         d: list, e: set, f: dict):
    pass


def test_a() -> [int]:
    return [1, 2, 3]


print(test_a())

import numpy as np

print(type(np.argmax([[1, 3, 2]])))
