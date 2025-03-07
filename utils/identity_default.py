from typing import TypeVar


K = TypeVar("K")
class IdentityDefault(dict[K,K]):
    def __missing__(self, key):
        return key
