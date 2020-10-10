# ------------------------------------------------------------------------------
# A module exemplifying style conventions for class and method definitions.
# ------------------------------------------------------------------------------
class ExampleClass:
    r"""Class description.

    Args:
        example_arg (type): description of __init__ argument
    """
    def __init__(self, example_arg):
        self.example_arg = example_arg

def example_method(arg_1, arg_2):
    r"""Description.

    Args:
        arg_1 (type): description
        arg_2 (type): description

    Returns (type): description

    Examples:
        >>> example_method(1, 2) --> 2
    """
    return arg_1*arg_2
