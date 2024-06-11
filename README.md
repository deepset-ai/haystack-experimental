[![PyPI - Version](https://img.shields.io/pypi/v/haystack-experimental.svg)](https://pypi.org/project/haystack-experimental)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/haystack-experimental.svg)](https://pypi.org/project/haystack-experimental)
[![Tests](https://github.com/deepset-ai/haystack-experimental/actions/workflows/tests.yml/badge.svg)](https://github.com/deepset-ai/haystack-experimental/actions/workflows/tests.yml)
[![Project release on PyPi](https://github.com/deepset-ai/haystack-experimental/actions/workflows/pypi_release.yml/badge.svg)](https://github.com/deepset-ai/haystack-experimental/actions/workflows/pypi_release.yml)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

# Haystack experimental package

The `haystack-experimental` package provides Haystack users with access to experimental features without immediately
committing to their official release. The main goal is to gather user feedback and iterate on new features quickly.

## Installation

For simplicity, every release of `haystack-experimental` will ship all the available experiments at that time. To
install the latest experimental features, run:

```sh
$ pip install -U haystack-experimental
```

> [!IMPORTANT]
> The latest version of the experimental package is only tested against the latest version of Haystack. Compatibility
> with older version of Haystack is not guaranteed.


## Experiments lifecycle

Any experimental feature will be removed from `haystack-experimental` after a period of 3 months. After this time,
the experiment will be either:
- Incorporated in Haystack and released with the next minor,
- Released as a Core Integration,
- Dropped.

## Experiments catalog

The latest version of the package contains the following experiments:

| Name                     | Type                    | Experiment end date |
| ------------------------ | ----------------------- | ------------------- |
| [`EvaluationHarness`][1] | Evaluation orchestrator | August 2024         |

[1]: https://github.com/deepset-ai/haystack-experimental/tree/main/haystack_experimental/evaluation/harness

## Usage

Experimental new features can be imported like any other Haystack integration package:

```python
from haystack.dataclasses import ChatMessage
from haystack_experimental.components.generators import FoobarGenerator

c = FoobarGenerator()
c.run([ChatMessage.from_user("What's an experiment? Be brief.")])
```

Experiments can also override existing Haystack features. For example, users opt-in for an experimental type of
`Pipeline` by just changing the usual import:

```python
# from haystack import Pipeline
from haystack_experimental import Pipeline

pipe = Pipeline()
# ...
pipe.run(...)
```

## Documentation

Documentation for `haystack-experimental` can be found [here](https://docs.haystack.deepset.ai/reference/haystack-experimental-api).

## Implementation

Experiments should try to replicate the namespace of the core package. For example, a new generator:

```python
# in haystack_experimental/components/generators/foobar.py

from haystack import component


@component
class FoobarGenerator:
    ...

```

When the experiment overrides an existing feature, the new symbol should be created at the same path in the experimental
package. This new symbol will override the original in `haystack-ai`: for classes, with a subclass and for bare
functions, with a wrapper. For example:

```python
# in haystack_experiment/src/haystack_experiment/core/pipeline/pipeline.py

from haystack.core.pipeline import Pipeline as HaystackPipeline


class Pipeline(HaystackPipeline):
	# Any new experimental method that doesn't exist in the original class
	def run_async(self, inputs) -> Dict[str, Dict[str, Any]]:
		...

	# Existing methods with breaking changes to their signature, like adding a new mandatory param
    def to_dict(new_param: str) -> Dict[str, Any]:
        # do something with the new parameter
        print(new_param)
        # call the original method
        return super().to_dict()

```

## Contributing

Direct contributions to `haystack-experimental` are not expected, but Haystack maintainers might ask you to move here
a pull request that you originally opened on https://github.com/deepset-ai/haystack.