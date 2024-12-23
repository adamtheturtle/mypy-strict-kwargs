|Build Status| |codecov| |PyPI|

mypy-strict-kwargs
==================

Enforce using keyword arguments where possible.
In the same spirit as a formatter - think ``black`` or ``ruff format`` - this lets you stop spending time discussing whether a particular function call should use keyword arguments.

For example, if we have a function which takes two regular argument, there are three ways to call it.
With this plugin, ``mypy`` will only accept the form where keyword arguments are used.

.. code-block:: python

   """Showcase errors when calling a function without naming the arguments."""


   def add(a: int, b: int) -> int:
       """Add two numbers."""
       return a + b


   add(a=1, b=2)  # With this plugin, mypy will only accept this form
   add(1, 2)  # type: ignore[misc]
   add(1, b=2)  # type: ignore[misc]

Installation
------------

.. code-block:: shell

   pip install mypy-strict-kwargs

This is tested on Python |minimum-python-version|\+.

Configure ``mypy`` to use the plugin by adding the plugin to your `mypy configuration file <https://mypy.readthedocs.io/en/stable/config_file.html>`_.

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

.. |Build Status| image:: https://github.com/adamtheturtle/mypy-strict-kwargs/actions/workflows/ci.yml/badge.svg?branch=main
   :target: https://github.com/adamtheturtle/mypy-strict-kwargs/actions
.. |codecov| image:: https://codecov.io/gh/adamtheturtle/mypy-strict-kwargs/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/adamtheturtle/mypy-strict-kwargs
.. |PyPI| image:: https://badge.fury.io/py/mypy-strict-kwargs.svg
   :target: https://badge.fury.io/py/mypy-strict-kwargs
.. |minimum-python-version| replace:: 3.12
