from typing_extensions import List


class ConversionsError(Exception):
    def __init__(self, message: str, conversion_exceptions: List[Exception]) -> None:
        super().__init__(message)
        self.conversion_exceptions = conversion_exceptions
