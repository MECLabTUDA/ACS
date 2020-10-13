# ------------------------------------------------------------------------------
# Introspection function to create object instances from string arguments.
# ------------------------------------------------------------------------------

def introspect(class_path):
    r"""Creates a class dynamically from a class path."""
    if isinstance(class_path, str):
        class_path = class_path.split('.')
    class_name = class_path[-1]
    module_path = class_path[:-1]
    module = __import__('.'.join(module_path))
    for m in module_path[1:]:
        module = getattr(module, m)
    module = getattr(module, class_name)
    return module