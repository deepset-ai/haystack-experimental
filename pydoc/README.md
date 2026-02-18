# :ledger: Looking for the docs?

You can find Haystack's documentation at https://docs.haystack.deepset.ai/.

# API Reference

We use [haystack-pydoc-tools](https://github.com/deepset-ai/haystack-pydoc-tools) to create Markdown files from the docstrings in our code. There is a [Github workflow](https://github.com/deepset-ai/haystack-experimental/blob/main/.github/workflows/docusaurus_sync.yml) that regenerates the API reference at release time.

If you want to generate a new Markdown file for a new Haystack module, create a `.yml` file in `pydoc` which configures how haystack-pydoc-tools will generate the page and commit it to main.

All the updates to API reference get pushed to documentation when a new version is released.

### Configuration

For configuration details, see the [haystack-pydoc-tools documentation](https://github.com/deepset-ai/haystack-pydoc-tools/blob/main/README.md#configuration) or inspect the existing `.yml` files in `pydoc`.
