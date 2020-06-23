from typing import List


class BaseConfig:
    @staticmethod
    def _get(value, default):
        if value is None:
            return default
        return value

    def _get_float(self, value: float, default: float) -> float:
        return self._get(value, default)

    def _get_int(self, value: int, default: int) -> int:
        return self._get(value, default)

    def _get_str(self, value: str, default: str) -> str:
        return self._get(value, default)

    def _get_list(self, value: List, default: List) -> List:
        return self._get(value, default)

    def _get_list_of_int(self, value: List[int], default: List[int]) -> List[int]:
        return self._get(value, default)

    def _get_list_of_float(self, value: List[float], default: List[float]) -> List[float]:
        return self._get(value, default)

    def _get_list_of_list_int(self, value: List[List[int]], default: List[List[int]]) -> List[List[int]]:
        return self._get(value, default)

    def _get_list_of_str(self, value: List[str], default: List[str]) -> List[str]:
        return self._get(value, default)

    def _get_bool(self, value: bool, default: bool) -> bool:
        return self._get(value, default)
