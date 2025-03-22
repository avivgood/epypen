from src.main import converted


def test_support_for_all_argument_types():
    @converted
    def _test(foo1, foo2, /, foo3, foo4, *, foo5, foo6, **foo8):
        pass

    _test(1, 2, 5, foo4=6, foo5=7, foo6=10, foo8=4)


def test_results():
    @converted
    def _sum(a, b):
        return a + b

    assert _sum(1, 2) == 3


def test_empty_call_to_the_decorator_factory():
    @converted()
    def _cast(x: int, y: int) -> int:
        return str(x + y)

    assert _cast("2", "3") == 5
