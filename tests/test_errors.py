import pytest
from src.exceptions import ConversionsError
from src.main import converted


def test_no_param_conversions_failure():
    def always_fail(typ, obj):
        raise ValueError()

    def always_fail2(typ, oj):
        raise TypeError()

    @converted(parameter_conversions=[always_fail, always_fail2])
    def foo(x, y):
        pass

    with pytest.raises(ConversionsError) as err:
        foo(1, 2)

    errs = err.value.conversion_exceptions
    assert len(errs) == 2
    assert isinstance(errs[0], ValueError)
    assert isinstance(errs[1], TypeError)


def test_no_return_type_conversions_failure():
    def always_fail(typ, obj):
        raise ValueError()

    def always_fail2(typ, oj):
        raise TypeError()

    @converted(return_value_conversions=[always_fail, always_fail2])
    def foo(x, y):
        pass

    with pytest.raises(ConversionsError) as err:
        foo(1, 2)

    errs = err.value.conversion_exceptions
    assert len(errs) == 2
    assert isinstance(errs[0], ValueError)
    assert isinstance(errs[1], TypeError)
