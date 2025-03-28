"""
``mypy`` plugin to enforce strict keyword arguments.
"""

import tomllib
from collections.abc import Callable
from configparser import ConfigParser
from functools import partial
from pathlib import Path
from typing import Any

from mypy.nodes import ArgKind
from mypy.options import Options
from mypy.plugin import FunctionSigContext, MethodSigContext, Plugin
from mypy.types import CallableType


def _parse_toml(config_file: Path) -> dict[str, Any] | None:
    """Returns a dict of config keys to values.

    Returns ``None`` if the config file is not a TOML file.
    """
    if config_file.suffix != ".toml":
        return None

    with config_file.open(mode="rb") as rf:
        return tomllib.load(rf)


def _transform_signature(
    ctx: FunctionSigContext | MethodSigContext,
    fullname: str,
    *,
    ignore_builtins: bool,
) -> CallableType:
    """
    Transform positional arguments to keyword-only arguments.
    """
    original_sig: CallableType = ctx.default_signature
    new_arg_kinds: list[ArgKind] = []

    star_arg_indices = [
        index
        for index, kind in enumerate(iterable=original_sig.arg_kinds)
        if kind == ArgKind.ARG_STAR
    ]

    first_star_arg_index = star_arg_indices[0] if star_arg_indices else None

    # Some methods get called with a positional argument that we do not supply.
    skip_first_argument_suffixes = (
        # Gets called when an instance of the class is called.
        ".__call__",
        # Descriptor attribute access
        ".__get__",
        # Descriptor attribute assignment
        ".__set__",
    )
    skip_first_argument = fullname.endswith(skip_first_argument_suffixes)

    skip_second_argument_suffixes = (
        # Descriptor attribute access.
        # The second argument is the instance of the class.
        ".__get__",
        # Descriptor attribute assignment.
        # The second argument is the value to be assigned.
        ".__set__",
    )

    skip_second_argument = fullname.endswith(skip_second_argument_suffixes)

    for index, (kind, name) in enumerate(
        iterable=zip(
            original_sig.arg_kinds,
            original_sig.arg_names,
            strict=True,
        )
    ):
        if skip_first_argument and index == 0:
            new_arg_kinds.append(kind)
            continue

        if skip_second_argument and index == 1:
            new_arg_kinds.append(kind)
            continue

        # If name is None, it is a positional-only argument; leave it as is
        is_positional_only = name is None
        should_ignore = ignore_builtins and fullname.startswith("builtins.")
        if is_positional_only or should_ignore:
            new_arg_kinds.append(kind)

        # Transform positional arguments that can also be keyword arguments
        elif kind == ArgKind.ARG_POS:
            if first_star_arg_index is None or index > first_star_arg_index:
                new_arg_kinds.append(ArgKind.ARG_NAMED)
            else:
                new_arg_kinds.append(kind)
        elif kind == ArgKind.ARG_OPT:
            if first_star_arg_index is None or index > first_star_arg_index:
                new_arg_kinds.append(ArgKind.ARG_NAMED_OPT)
            else:
                new_arg_kinds.append(kind)
        else:
            new_arg_kinds.append(kind)

    return original_sig.copy_modified(arg_kinds=new_arg_kinds)


class _KeywordOnlyPluginConfig:
    """A mypy plugin config holder.

    Attributes:
        ignore_builtins: Whether to ignore built-in functions.
    """

    __slots__ = ("ignore_builtins",)
    ignore_builtins: bool

    def __init__(self, options: Options) -> None:
        if options.config_file is None:  # pragma: no cover
            return

        toml_config = _parse_toml(config_file=Path(options.config_file))
        if toml_config is not None:
            config = toml_config.get("tool", {})
            config = config.get("mypy_strict_kwargs", {})
            for key in self.__slots__:
                setting = config.get(key, False)
                if not isinstance(setting, bool):
                    msg = (
                        f"Configuration value must be a boolean for key: {key}"
                    )
                    raise TypeError(msg)
                setattr(self, key, setting)
        else:
            plugin_config = ConfigParser()
            plugin_config.read(filenames=options.config_file)
            for key in self.__slots__:
                setting = plugin_config.getboolean(
                    section="mypy-strict-kwargs",
                    option=key,
                    fallback=False,
                )
                setattr(self, key, setting)

    def to_data(self) -> dict[str, Any]:
        """
        Returns a dict of config names to their values.
        """
        print(str(1))
        return {key: getattr(self, key) for key in self.__slots__}


class KeywordOnlyPlugin(Plugin):
    """
    A plugin that transforms positional arguments to keyword-only arguments.
    """

    def __init__(self, options: Options) -> None:
        """
        Configure the plugin.
        """
        super().__init__(options=options)
        if options.config_file is None:  # pragma: no cover
            return
        self._plugin_config = _KeywordOnlyPluginConfig(options=options)
        self._plugin_data = self._plugin_config.to_data()

    def get_function_signature_hook(
        self,
        fullname: str,
    ) -> Callable[[FunctionSigContext], CallableType] | None:
        """
        Transform positional arguments to keyword-only arguments.
        """
        return partial(
            _transform_signature,
            fullname=fullname,
            ignore_builtins=self._plugin_data["ignore_builtins"],
        )

    def get_method_signature_hook(
        self,
        fullname: str,
    ) -> Callable[[MethodSigContext], CallableType] | None:
        """
        Transform positional arguments to keyword-only arguments.
        """
        return partial(
            _transform_signature,
            fullname=fullname,
            ignore_builtins=self._plugin_data["ignore_builtins"],
        )


def plugin(version: str) -> type[KeywordOnlyPlugin]:
    """
    Plugin entry point.
    """
    del version  # to satisfy vulture
    return KeywordOnlyPlugin
