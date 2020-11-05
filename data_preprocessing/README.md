# Data processing

To correcly install dependencies to generate poincare embeddings:
```
python -m venv .venv
source .venv/bin/activate
xargs -L 1 pip install < requirements.txt
```
Note: the setup.py in poincare-embeddings pakage requires some dependencies and we need to install everything sequentially rather than `pip install -r requirements.txt`.

Now you can use the poincare-embeddings package by
```
python $VIRTUAL_ENV/src/poincare/embed.py
```