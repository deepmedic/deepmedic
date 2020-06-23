from typing import List


def _pp_config_repr(config: "BaseConfig", indent: int = 1):
    """Pretty print config with correct indents"""
    s = config.__class__.__name__
    s += "(\n"
    for i, slot in enumerate(config.__slots__):
        if i > 0:
            s += ", \n"
        for _ in range(indent):
            s += "\t"
        s += slot
        s += "="
        val = getattr(config, slot)

        if isinstance(val, BaseConfig):
            s += _pp_config_repr(val, indent + 1)
        elif isinstance(val, list) and len(val) > 0:
            if isinstance(val[0], BaseConfig):
                s += "[\n"
                for _ in range(indent + 1):
                    s += "\t"
                for idx, item in enumerate(val):
                    s += _pp_config_repr(item, indent + 2)
                    if idx < len(val) - 1:
                        s += ",\n"
                        for _ in range(indent + 1):
                            s += "\t"
                    else:
                        s += "\n"
                        for _ in range(indent):
                            s += "\t"

                s += "]"
            else:
                s += repr(getattr(config, slot))
        else:
            s += repr(getattr(config, slot))
    s += "\n"
    for i in range(indent - 1):
        s += "\t"
    s += ")"
    return s


class BaseConfig:
    __slots__ = []

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

    def __eq__(self, other):
        for attr in self.__slots__:
            if getattr(other, attr) != getattr(self, attr):
                return False
        return True

    def __repr__(self):
        return _pp_config_repr(self)
