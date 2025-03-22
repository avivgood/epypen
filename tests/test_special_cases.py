import asyncio
import pytest
from src.main import converted


@pytest.mark.asyncio
async def test_async_support():
    @converted
    async def async_func(time: float):
        await asyncio.sleep(time)

    await async_func("0.01")


def test_instance_method_support():
    class A:
        def __init__(self, x):
            self.x = x

        @converted
        def cast(self, y: int) -> int:
            return str(self.x + y)

    a = A(2)
    assert a.cast("3") == 5


def test_class_method_support():
    class A:
        @classmethod
        @converted
        def cast(cls, x: int) -> int:
            return str(x)

    assert A.cast("5") == 5


def test_static_method_support():
    class A:
        @staticmethod
        @converted
        def cast(x: int) -> int:
            return str(x)

    assert A.cast("5") == 5


def test_unannotated_return_type():
    @converted
    def unannotated_return(x: int):
        return str(x)

    # Should work without type conversion since return type is not annotated
    assert unannotated_return("5") == "5"
