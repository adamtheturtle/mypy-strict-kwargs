|Build Status| |codecov| |PyPI|

mypy-strict-kwargs
==================

Enforce using keyword arguments where possible.

For example, if we have a function like this:

.. code-block:: python

   def func(a: int, b: int) -> None: ...

then we can call it in multiple ways:

.. code-block:: python

   func(1, 2)
   func(a=1, b=2)  # With this plugin, mypy will only accept this form
   func(1, b=2)

Installation
------------

.. code-block:: shell

   pip install mypy-strict-kwargs

This is tested on Python |minimum-python-version|\+. Get in touch with
``adamdangoor@gmail.com`` if you would like to use this with another
language.

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
