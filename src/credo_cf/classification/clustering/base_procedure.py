from abc import abstractmethod
from typing import Any, List


class BaseProcedure:
    @abstractmethod
    def procedure(self, stack: Any) -> Any:
        pass


def execute_chain(chain: List[BaseProcedure], data: Any) -> Any:
    current = data
    for bp in chain:
        current = bp.procedure(current)
    return current
