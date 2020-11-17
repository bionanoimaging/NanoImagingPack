# How to create the READ-THE-DOCS ?

Inspired by this here https://coderefinery.github.io/documentation/05-rtd/
Install

```
pip install commonmark recommonmark
pip install sphinx_rtd_theme
```



Go to the base folder of the NIP toolboc and enter:

```
sphinx-build doc _builddoc
```

The documentation is compiled into the folder ```_builddoc```

