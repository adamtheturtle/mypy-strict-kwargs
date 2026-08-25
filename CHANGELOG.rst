Changelog
=========

.. towncrier release notes start

2026.08.25
----------

- Stop reporting a decorator applied to an overloaded callable, which is
  applied by position and has no keyword form.

- Call mypy's own attribute hooks instead of shadowing them, so that the
  value type behind ``SomeEnum.value`` is no longer lost.

2026.08.20.1
------------

- Report positional calls to overloaded functions as call-arg errors
  instead of a generic call-overload message.

- Stop reformatting the plugin test cases, whose assertions depend on the
  line numbers of the source they embed.

- Type check real projects with the plugin on a schedule, so that a crash
  cannot reach a release.

- Compose with mypy's own signature hooks instead of shadowing them, so
  checks such as ``dataclasses.replace`` keep working and TypedDict methods
  stop reporting a misleading second error.

- Capture standard error when checking real projects for crashes, which is
  where the crash is reported.

2026.08.20
----------

- Follow an assignment through ``nonlocal`` to the nearest enclosing scope which
  binds that name, instead of to every enclosing scope.

- Recognize a ``Union[...]`` annotation and a type alias to a union when
  finding the length of a fixed tuple.

- Stop reporting positional arguments for ``super()`` calls whose written
  starting type cannot be resolved, such as a class alias imported from
  another module.

- Follow the length of a sequence bound by an assignment expression, and by
  an assignment which follows a ``global`` declaration, and stop a
  ``global`` declaration from letting an enclosing scope's length show
  through.

- Stop an assignment through ``nonlocal`` in a nested function from
  discarding what is known about a module-level name.

2026.08.19
----------

- Stop requiring keyword arguments for ``__aexit__`` so that ``async with``
  statements type check.

- Check positional arguments supplied to ``super()`` methods by unpacking
  set, dictionary, string and bytes literals.

- Check positional arguments supplied to ``super()`` methods by unpacking
  a name or expression whose sequence length is statically known.

- Check positional arguments supplied to ``super()`` methods from
  declared types, such as the return type of an annotated function, an
  annotated attribute, a ``cast()`` and a context manager.

- Check positional arguments supplied to ``super()`` members which are
  not methods, such as assigned functions, static and class methods
  wrapping arbitrary callables, and callable objects.

- Stop reporting false positives for ``super()`` calls where a name bound
  by an enclosing scope is rebound nearer to the call.

- Resolve type aliases, ``NamedTuple`` types, ``NewType`` types and
  ``Annotated`` wrappers when finding the length of a fixed tuple
  parameter used in a ``super()`` call.

- Expand PEP 646 unpacks when counting the length of fixed tuple
  annotations used in ``super()`` calls.

- Check positional arguments supplied to ``super()`` methods from names
  bound by iterating over annotated iterables of fixed tuples, and by
  match captures.

- Stop requiring keyword arguments for special methods which Python invokes
  implicitly, such as ``__rdivmod__`` used by ``divmod()``.

- Start ``super()`` method lookup at the right entry when the explicit
  first argument is a qualified name or a class alias.

- Print the name used to check a ``super()`` method call in debug output, so
  that it can be added to ``ignore_names``.

- Leave names found while checking type stubs out of debug output.

- Check positional arguments supplied to ``super()`` methods through nested
  spreads of known length.

- Check positional arguments which ``functools.partial`` and
  ``functools.partialmethod`` bind on the callable they wrap.

- Report a configuration error instead of silently splitting a string
  ``ignore_names`` value into characters.

- Report a configuration error for non-string ``ignore_names`` entries.

- Report a configuration error instead of crashing when the
  ``mypy_strict_kwargs`` configuration section is not a table.

- Report a configuration error for non-boolean ``debug`` values instead of
  coercing them.

- Stop comprehension targets from shadowing names in the leftmost iterable,
  which is evaluated in the enclosing scope.

- Treat a ``nonlocal`` declaration as an alias of an enclosing binding
  rather than as a new binding.

- Stop requiring a keyword argument for the modulo parameter of
  ``__rpow__``, which a ternary ``pow()`` supplies by position.

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
