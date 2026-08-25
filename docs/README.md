# bimets documentation

The documentation is organized into three parts:

- tutorials for learning the library through complete examples;
- reference pages for exact signatures, parameters, and return types;
- explanations of design decisions, compatibility, validation, and solver
  behavior.

## Tutorials

The [tutorials](tutorial/README.md) follow a progressive learning path, from working with time series to running complete econometric workflows:

- [Getting started](tutorial/01.getting-started.md)
- [Time series, access, and indexing](tutorial/02.timeseries.md)
- [Manipulating time series](tutorial/03.manipulating-timeseries.md)
- [Your first model](tutorial/04.first-model.md)
- [Complete tutorial and guide index](tutorial/README.md)

## API reference

The [API reference](reference/README.md) contains static pages generated from
the NumPy-style docstrings of public functions, classes, methods, and
properties. The [public API inventory](reference/api.md) separately maps
canonical Python names to BIMETS R compatibility aliases.

## Explanations

The [explanation index](explanation/README.md) collects conceptual background,
differences from BIMETS R, solver design, and numerical conformance evidence.

- [Main differences from BIMETS R](explanation/migration-from-r.md)
- [Solver architecture and execution strategies](explanation/solver-strategies.md)
- [Compatibility and numerical validation](explanation/conformance.md)

[Back to the project README](../README.md)
