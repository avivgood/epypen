from typing_extensions import List


class EpypenException(Exception):
    pass


class ConversionsError(EpypenException):
    def __init__(self, message: str, conversion_exceptions: List[Exception]) -> None:
        super().__init__(message)
        self.conversion_exceptions = conversion_exceptions
