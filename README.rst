|Build Status| |codecov| |PyPI|

mypy-strict-kwargs
==================

Enforce using keyword arguments where possible.

For example, if we have a function which takes two regular argument, there are three ways to call it.
With this plugin, ``mypy`` will only accept the form where keyword arguments are used.

.. code-block:: python

   """Showcase errors when calling a function without naming the arguments."""


   def add(a: int, b: int) -> int:
       """Add two numbers."""
       return a + b


   add(a=1, b=2)  # With this plugin, ``mypy`` will only accept this form
   add(1, 2)  # type: ignore[misc]
   add(1, b=2)  # type: ignore[misc]

Why?
----

* In the same spirit as a formatter - think ``black`` or ``ruff format`` - this lets you stop spending time discussing whether a particular function call should use keyword arguments.
* Sometimes positional arguments are best at first, and then more and more are added and code becomes unclear, without anyone stopping to refactor to keyword arguments.
* The type checker gives better errors when keyword arguments are used.
  For example, with positional arguments, you may see, ``Argument 5 to "add" has incompatible type "str"; expected "int"``.
  This requires that you count the arguments to see which one is wrong.
  With named arguments, you get ``Argument "e" to "add" has incompatible type "str"; expected "int"``.

Installation
------------

.. code-block:: shell

   pip install mypy-strict-kwargs

This is tested on Python |minimum-python-version|\+.

Configure ``mypy`` to use the plugin
------------------------------------

Add the plugin to your `mypy configuration file <https://mypy.readthedocs.io/en/stable/config_file.html>`_:

``.ini`` files:

.. code-block:: ini

   [mypy]
   plugins = mypy_strict_kwargs

``.toml`` files:

.. code-block:: toml

   [tool.mypy]

   plugins = [
       "mypy_strict_kwargs",
   ]

Ignoring functions
------------------

You can ignore functions by adding configuration to ``pyproject.toml``.

.. code-block:: toml

   [tool.mypy_strict_kwargs]
   ignore_names = ["main.func", "builtins.str"]

This is useful especially for builtins which can look strange with keyword arguments.
For example, ``str(object=1)`` is not idiomatic.

To find the name of a function to ignore, set the following configuration:

.. code-block:: toml

   [tool.mypy_strict_kwargs]
   debug = true

Then run ``mypy`` and look for the debug output.

.. |Build Status| image:: https://github.com/adamtheturtle/mypy-strict-kwargs/actions/workflows/ci.yml/badge.svg?branch=main
   :target: https://github.com/adamtheturtle/mypy-strict-kwargs/actions
.. |codecov| image:: https://codecov.io/gh/adamtheturtle/mypy-strict-kwargs/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/adamtheturtle/mypy-strict-kwargs
.. |PyPI| image:: https://badge.fury.io/py/mypy-strict-kwargs.svg
   :target: https://badge.fury.io/py/mypy-strict-kwargs
.. |minimum-python-version| replace:: 3.11
