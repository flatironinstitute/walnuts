# `walnutpie`: Adaptive Walnuts in Python and C++

This is a C++ implementation and Python wrapper of the following [Hamiltonian Monte
Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) (HMC) samplers.

* [Walnuts](https://arxiv.org/abs/2506.18746)
* Adaptive Walnuts (continuous form of [Nutpie](https://github.com/pymc-devs/nutpie)-style adaptation)

## Documentation

Documentation for `walnutpie` can be found on [Github Pages](https://flatironinstitute.github.io/walnuts/latest/).

## Using walnutpie from Python

`walnutpie` is distributed on PyPI and can be installed with

```bash
pip install walnutpie
```

For more information, consult [the documentation](https://flatironinstitute.github.io/walnuts/latest/install.html).

## Using walnutpie in a C++ project

The `walnutpie` library is header-only and only requires
[Eigen](https://gitlab.com/libeigen/eigen) (also header-only) to use.

If your project uses CMake, you can depend on our
`walnutpie` library target. If not, any method of adding the `include/`
folder of this repository to your build system's include paths should suffice
as long as you also provide Eigen yourself. See the [examples/
directory](./examples/) for more on usage.

## For developers

Interested in editing the code or contributing to `walnutpie`? Consult [CONTRIBUTING.md](./CONTRIBUTING.md)

## Licensing

The project is distributed under the following licenses.

* Code: [MIT License](https://opensource.org/license/mit)
* Documentation: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)
