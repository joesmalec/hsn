# HSN — Hierarchical Synaptic Network

HSN is a research proposal for a sparse, routed, continually learning recurrent network. This
repository implements [`HSN_Implementation_Plan.md`](HSN_Implementation_Plan.md), which turns the
white paper into phases, acceptance gates (G0–G6) and chunks (`P<phase>.<n>`); one chunk is one pull
request by one implementer. No HSN results exist yet; every threshold in the plan is a proposed
decision rule, not a performance promise.

**Stack.** Python/PyTorch trains and runs experiments (gates G0–G4): the package `hsn` under
`python/`. Rust serves (gates G5–G6): runtime, streaming loader, replay store, registry and signing in
the cargo workspace under `rust/`, exposed to Python as `hsn_rs` through PyO3. GPU kernels are written
once per vendor (one CUDA source, one Metal source) and used by both hosts.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) ≥ 0.13.6 (`brew install uv`). uv installs the pinned Python 3.12
  itself (`.python-version`).
- [rustup](https://rustup.rs) (`brew install rustup`, then put `$(brew --prefix rustup)/bin` on your
  PATH). The toolchain is pinned in `rust-toolchain.toml`; `rustup toolchain install` installs it.
- No GPU, CUDA toolkit or Metal toolchain is needed for the base install: everything here runs on
  `profile:cpu`.

## Commands every implementer uses

```bash
uv sync --extra dev                                    # Python env in .venv; installs hsn editable
uv run maturin develop --uv -m rust/hsn-py/Cargo.toml  # build hsn_rs into the env
cargo test --workspace                                 # Rust tests
uv run pytest -q                                       # Python tests (CPU profile by default)
uv run python scripts/build_kernels.py --target cuda|metal|all|none   # L2 kernels; prints "no toolchain" without one
uv run pre-commit run --all-files                      # ruff, ruff-format, mypy --strict, rustfmt, clippy -D warnings
```

> **Sync order matters.** `uv sync` makes `.venv` match `uv.lock` exactly and removes anything else,
> including the `hsn_rs` module that `maturin develop` installed. Run `uv sync` first and
> `maturin develop` second. After any later `uv sync`, either pass `--inexact` or run
> `maturin develop` again. `uv run …` never removes it.

Bare `cargo` commands build PyO3 against `.venv/bin/python` (see `.cargo/config.toml`), so run
`uv sync` before `cargo test`. A test that needs a GPU is marked `@pytest.mark.cuda` or
`@pytest.mark.mps` and skips when the device is absent.

## Hardware profiles

Chunks never name a device; they refer to a profile (plan §3.1 and Appendix E):

| Profile | Hardware | Role |
| --- | --- | --- |
| `profile:rtx3060` | NVIDIA RTX 3060 12 GB (CUDA, sm_86) | low-VRAM reference; day-one development |
| `profile:a100` | A100 80 GB / H100 (CUDA) | confirmation runs |
| `profile:dgx_spark` | NVIDIA DGX Spark, GB10 (aarch64 + CUDA, sm_121) | large configurations; aarch64 builds |
| `profile:m_series` | Apple M-series, macOS 15+ (Metal via PyTorch MPS) | development; Metal kernels; parity tests |
| `profile:cpu` | any x86_64 or aarch64 | numerical oracle; CI on every PR |

## PyTorch wheels for GPUs and aarch64

`pyproject.toml` pins `torch==2.15.0` from PyPI and deliberately carries **no** index URL, so the
base install works on a laptop without a GPU. On Linux x86_64 the PyPI wheel is the CUDA 13.2
(`cu132`) build; official aarch64 Linux wheels (DGX Spark) are also on PyPI; macOS uses the MPS backend
of the PyPI wheel and needs macOS 15 or newer. Other variants come from PyTorch's own indexes, for
example:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu132 torch==2.15.0   # CUDA 13.2, x86_64
uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0     # CPU-only Linux
```

Optional extras: `uv sync --extra cuda` (pynvml, for the energy meter) and `--extra viz` (matplotlib).

## Continuous integration

- `.github/workflows/ci.yml` runs on every pull request and push to `main`: `ubuntu-cpu`
  (ubuntu-latest, x86_64) and `macos-arm64-cpu` (macos-26, arm64) run the commands above on the CPU
  profile. Hosted runners cannot execute CUDA or Metal kernels; `build_kernels.py` reports
  `no toolchain` there.
- `.github/workflows/nightly.yml` holds the self-hosted jobs `rtx3060`, `dgx_spark` and `m_series`.
  Each is skipped until the repository variable `HSN_SELF_HOSTED_RUNNERS` (comma-separated labels,
  e.g. `rtx3060,m_series`) names a registered runner that carries that label.

## Contributing

One chunk per pull request, titled `P<phase>.<n>: <chunk name>`; the description lists files, tests,
measured numbers and any deviation from the chunk text with its reason (plan §0.4 and §4.3). All tests
must pass on the CPU profile and pre-commit must be clean. Measured, not assumed: target numbers from
configs are never reported as results.

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Local terminal chat

Install with `uv sync --extra dev --extra chat` (then rebuild `hsn_rs` with the
maturin command above if needed). Launch from the repository:

```bash
uv run hsn chat --run-dir /absolute/path/to/training/run
uv run hsn chat --checkpoint /path/step_00001000.pt --config /path/config.resolved.yaml --device cpu
```

The first command selects the highest numbered checkpoint. The second selects an exact
checkpoint. Resolved configs must match checkpoint metadata. Training data is not needed
at inference time. The optional UI dependency is only loaded by `hsn chat`.

Enter inserts a newline; Ctrl+Enter or the Send button sends. Escape stops generation;
Ctrl+N starts a fresh conversation (while generating, it requests stop first). Ctrl+E
exports JSON to a new timestamped file in the current directory. Ctrl+F toggles filler
filtering; Ctrl+Q exits. Exports retain original input and the filtered model input.
There is no automatic saving or session restore.

Standalone `um`, `uh`, and `erm` are removed before inference by default. Quoted text and
backtick spans are preserved. This deliberately avoids general stop-word removal: words
such as “not,” “can,” and “like” carry meaning. Use `--no-filter-fillers` when exact input
matters, especially for code or raw synthetic tasks. The status bar reports bytes saved;
HSN uses byte tokens, not word tokens. Input that becomes empty is rejected.

Generation options: `--temperature` (0.8; zero is greedy), `--top-p` (0.95), `--seed` (0),
`--context-id` (0), `--max-bytes` (512), and `--device auto|cpu|cuda|mps`.
Auto prefers CUDA, then MPS, then CPU. Sampling uses a session-local CPU generator;
reproducibility is within the same model/device, not a cross-device guarantee.

The default `plain-v1` format encodes `User: <message>\nAssistant: ` as UTF-8 and separates
subsequent turns with a newline. Generation stops before `\nUser:`, at the byte limit, or
when cancelled. `--format raw` feeds exact prompt continuations without role delimiters
(use `--no-filter-fillers` to disable preprocessing too). No EOS token is invented.
Future conversation training must use the same serialization in `hsn.chat.session.serialize`.
Current synthetic-task checkpoints do not imply natural-language conversation capability.

Inference retains recurrent state but never updates model weights. Reset clears that state.
An inference error or cancellation during prompt ingestion requires a reset. Generation
cancellation retains the partial reply. Malformed UTF-8 is displayed with replacement
characters; terminal controls are escaped. The current implementation uses Python/PyTorch
HSN inference, not the future Rust service, and supports the 256-byte vocabulary only.
