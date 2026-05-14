# vsim

Loss-aware sampling for Stim circuits (surface-code workflows).


## Install

```bash
uv sync --group dev        # editable dev env (pytest, notebooks, …)
# or
pip install vsim
```

Run tests:

```bash
uv run --group dev pytest
```

Regenerate the golden loss histogram snapshot (uses `devtools/legacy_loss_lib.py`):

```bash
uv run --group dev python devtools/regenerate_loss_snapshot.py
```
