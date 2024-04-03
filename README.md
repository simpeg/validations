# simpeg_validations

Validation of SimPEG code against analytic solutions and UBC-GIF modeling packages.

## How to build the website

The SimPEG validation websites was created using [Jupyter Book][jupyterbook].
In order to build it from its sources, you first need to [install
`jupyter-book`][install-jupyterbook] with `pip`:

```bash
pip install -U jupyter-book
```

or with `conda` (or `mamba`):

```bash
conda install -c conda-forge jupyter-book
```

Once you have `jupyter-book` installed, clone this repository to download the
sources:

```bash
git clone https://www.github.com/simpeg/validations
cd validations
```

You can now build the website by simply running:

```bash
jupyter-book build .
```

Finally, you can preview the website by opening the `_build/html/index.html`
with your web browser.

[jupyterbook]: https://jupyterbook.org
[install-jupyterbook]: https://jupyterbook.org/en/stable/start/overview.html#install-jupyter-book
