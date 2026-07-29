# quant-factors

Shared factor computation library for PureSaber quant research.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
quant-factors list
quant-factors compute --config configs/example.yaml
```

## Factors

- `momentum_20d`
- `volatility_20d`
- `mean_reversion_z_20d`
- `volume_surge_5d`
