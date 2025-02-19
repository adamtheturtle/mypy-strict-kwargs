from collections.abc import Callable


class A:
    def __call__(self) -> None:
        pass


class B:
    def __init__(self, c: Callable[[], None]) -> None:
        self.c = c


a = A()
b = B(c=a)
if (
    b.c is a
):  # pyright shows `reportUnnecessaryComparison` "Condition will always evaluate to False"
    print("Here")  # "Here" is printed
