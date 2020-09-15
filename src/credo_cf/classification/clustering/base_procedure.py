from abc import abstractmethod
from typing import Any, List


class BaseProcedure:
    """
    Interface of procedure object. Used in execute_chain.
    """

    @abstractmethod
    def procedure(self, stack: Any) -> Any:
        pass


def execute_chain(chain: List[BaseProcedure], data: Any) -> Any:
    """
    Execute ``procedure`` method form list of BaseProcedure objects in ``chain`` arg.
    The method get data as args and returns data that be used as data args in next object from list.

    :param chain: list of object to execute the procedure method for its
    :param data: data for first first object for chain
    :return: result from execution last object
    """
    current = data
    for bp in chain:
        current = bp.procedure(current)
    return current
