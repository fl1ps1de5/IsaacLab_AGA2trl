from typing import Any


# define because cannot use lambda with pytorch vmap etc
# modified from skrl.agents.torch.base.Agent._empty_preprocessor
def empty_preprocessor(input: Any, *args, **kwargs) -> Any:
    return input
