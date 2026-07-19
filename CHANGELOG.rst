Changelog
=========

.. towncrier release notes start

2026.07.19.1
------------

- Check positional arguments supplied to ``super()`` methods by unpacking
  list literals.

- Stop reporting false positives for ``super()`` methods called with a
  tuple literal that unpacks a variable-length ``*`` spread.

2026.07.19
----------

- Fix false positives for ``super()`` calls whose remaining positional
  arguments are consumed by ``*args``.

- Respect the explicit starting type when checking two-argument ``super()``
  calls.

- Check nested-class ``super()`` calls against the nested class's MRO instead
  of the enclosing class's MRO.

- Stop ``super()`` method lookup at the first MRO entry defining the member,
  matching Python's runtime attribute lookup.

- Check positional arguments supplied to ``super()`` methods by unpacking
  fixed-length tuples.

- Restrict protocol-method positional exemptions to implicit calls.

- Invalidate mypy's incremental cache when plugin configuration changes.

- Check ``super()`` calls to methods assigned through ``staticmethod()``.

2026.05.20.1
------------


2026.05.20
----------


2026.05.19
----------


- Drop Python 3.10 support (requires Python >=3.11).

2026.01.12
----------


- Add support for ``setup.cfg`` and ``mypy.ini`` configuration files.

2025.04.03
----------

2025.03.28
----------

2024.12.25
----------

2024.12.24
----------

2024.12.23.2
------------

2024.12.23.1
------------

2024.12.23
----------
