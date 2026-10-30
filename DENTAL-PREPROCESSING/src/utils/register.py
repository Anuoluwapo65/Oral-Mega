from typing import Callable, Dict 
import numpy as np   

REGISTRY = {
    "loaders": {},
    "transforms":{},
    "callbacks":{}}


def register(extension:str, registry: Dict[str, Callable]):
    def decorator(func: Callable):
        registry[extension] = func
        return func
    return decorator
