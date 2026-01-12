"""
``mypy`` plugin to enforce strict keyword arguments.
"""

import configparser
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

import tomli as tomllib
from mypy.nodes import ArgKind
from mypy.options import Options
from mypy.plugin import FunctionSigContext, MethodSigContext, Plugin
from mypy.types import CallableType


def _transform_signature(
    ctx: FunctionSigContext | MethodSigContext,
    fullname: str,
    *,
    ignore_names: list[str],
    debug: bool,
) -> CallableType:
    """
    Transform positional arguments to keyword-only arguments.
    """
    if debug:
        sys.stderr.write(f"DEBUG: mypy_strict_kwargs: {fullname}\n")

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
        should_ignore = fullname in ignore_names
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

    # See https://github.com/facebook/pyrefly/issues/1995.
    return original_sig.copy_modified(
        arg_kinds=new_arg_kinds,  # pyrefly: ignore[bad-argument-type]
    )


class KeywordOnlyPlugin(Plugin):
    """
    A plugin that transforms positional arguments to keyword-only arguments.
    """

    def __init__(self, options: Options) -> None:
        """Configure the plugin.

        This is not friendly to errors yet.
        """
        super().__init__(options=options)
        self._ignore_names: list[str] = []
        self._debug = False

        if options.config_file is None:
            return

        config_file = Path(options.config_file)
        if config_file.suffix == ".toml":
            with config_file.open(mode="rb") as rf:
                config_dictionary = tomllib.load(rf)

            tools = dict(config_dictionary.get("tool", {}))
            plugin_config = dict(tools.get("mypy_strict_kwargs", {}))
            self._ignore_names = list(plugin_config.get("ignore_names", []))
            self._debug = bool(plugin_config.get("debug", False))
        else:
            # Handle ``mypy.ini``, ``.mypy.ini``, ``setup.cfg``
            parser = configparser.ConfigParser()
            parser.read(filenames=config_file)

            if parser.has_section(section="mypy_strict_kwargs"):
                ignore_names_str = parser.get(
                    section="mypy_strict_kwargs",
                    option="ignore_names",
                    fallback="",
                )
                if ignore_names_str:
                    self._ignore_names = [
                        name.strip()
                        for name in ignore_names_str.split(sep=",")
                        if name.strip()
                    ]
                self._debug = parser.getboolean(
                    section="mypy_strict_kwargs",
                    option="debug",
                    fallback=False,
                )

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
            ignore_names=self._ignore_names,
            debug=self._debug,
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
            ignore_names=self._ignore_names,
            debug=self._debug,
        )


def plugin(version: str) -> type[KeywordOnlyPlugin]:
    """
    Plugin entry point.
    """
    del version  # to satisfy vulture
    return KeywordOnlyPlugin
