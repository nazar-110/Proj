# Contributing

Thanks for your interest in contributing!

## Development setup

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
ctest --output-on-failure
```

## Formatting

Run clang-format before submitting changes:

```bash
cmake --build . --target format
```

## Guidelines

- Keep kernels well-commented and include a baseline implementation.
- Add CPU reference updates if you add a new GPU kernel.
- Update documentation if you change CLI flags or build options.
