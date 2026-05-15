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

Fetch stim C++ sources used by the native extension (not committed to git):

```bash
python scripts/fetch_stim_header.py --version 1.15.0
```

`CMakeLists.txt` also auto-fetches these sources during extension builds if
`vendor/stim/` is missing.

Regenerate the golden loss histogram snapshot (uses `devtools/legacy_loss_lib.py`):

```bash
uv run --group dev python devtools/regenerate_loss_snapshot.py
```
