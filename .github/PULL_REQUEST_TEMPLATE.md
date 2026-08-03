## What

<!-- One change per PR. What does it do, and why? -->

## Numerics

- [ ] Bit-identical: `pytest tests/ -v` green, kernel gates included (paste the Metal gate summary if you touched a kernel)
- [ ] OR deliberate numerics change: behind an env flag, previous behavior recoverable, documented in the README tuning table, validated end-to-end

## Performance (if claimed)

<!-- Paired same-session A/B, baseline re-run as drift control. -->

- Machine: <!-- chip (base/Pro/Max/Ultra), RAM, macOS, mlx version -->
- Numbers: <!-- before / after / drift control; harness used (bench/...) -->

## Checklist

- [ ] `pytest tests/ -v` passes locally
- [ ] Docs updated where behavior or defaults changed
- [ ] Commits signed off (`git commit -s`)
