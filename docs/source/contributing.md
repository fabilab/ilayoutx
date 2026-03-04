# Contributing
## Suggestions
We keep track of general suggestions on sourcehut, please open a ticket there!
## Code / documentation / tests
ilayoutx is mainly developed on sourcehut, but we accept contributions from GitHub as well.

::::{tab-set}

:::{tab-item} SourceHut
Open a [ticket on SourceHut](https://todo.sr.ht/~iosonofabio/ilayoutx) to request features, report bugs, or show intention in contributing.

After an initial conversation, you will be requested to create a patch and submit it via email. The best way to contribute code from SourceHut is via patches sent to the mailing list. Read the official sourcehut docs for info.

:::

:::{tab-item} Github
Open an [issue on GitHub](https://github.com/fabilab/ilayoutx/issues) to request features, report bugs, or show intention in contributing.

The best way to contribute code from GitHub is via Pull Requests (PRs). For new contributors:

- Log into Github
- [Create a fork of ilayoutx](https://github.com/fabilab/ilayoutx).
- Clone this fork locally:

```
git clone https://github.com/YOUR-USERNAME/ilayoutx.git
```

- Create a local branch and switch to it (you can choose any branch name you fancy):

```
git checkout -b <branch name>
```

- Make your changes, using commits to document your work.

- Test your changes by writing and ensuring you pass unit tests via `pytest` (see below).

- Push your commits to your forked repository:

```
git push --set-upstream origin <branch name>
```

- From there, make the pull request against the ```main``` repository, including a clear title and a detailed description of what your code accomplishes.
:::

::::

### Local setup
Create a local virtual environment:
```
python -m venv .venv
```
Update pip (to support dependency groups):
```
.venv/bin/pip install --upgrade pip
```
Install maturin:
```
.venv/bin/pip install maturin
```
Develop the project in editable mode:
```
.venv/bin/maturin develop
```
Install test dependencies:
```
.venv/bin/pip install --groups test .
```

### Tests
If you have added code that needs testing, add tests. Ensure the existing test suite passes before submitting your changes.

To test from a local setup, run:
```
.venv/bin/pytest
```
and make sure all tests are either passing or skipped. If you'd like to request help with passing tests, write to the issue you opened previously on SourceHut/GitHub.
