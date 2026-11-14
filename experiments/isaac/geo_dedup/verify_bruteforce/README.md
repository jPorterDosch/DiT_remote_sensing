# Independent brute-force check of the geographic dedup artifacts

Written by a review agent on 2026-10-13 as a second, independent implementation of the
exclusion rule (own Web-Mercator tile bounds, own exact box-to-box great-circle gap: 0 on
overlap, else the minimum over the eight vertices of the point-to-box distance with the
meridian critical point; self-tested against 600-point-per-edge boundary sampling, worst
0.057 m). It reads the executed artifacts under `$STORE/geo_dedup/` and never calls
`eval.geo_dedup` for the decision itself (only to compare tile bounds in check 6).

    # from the repository root (-p short -q short, 4 CPUs, 64 GB, ~3.5 min):
    mkdir -p logs
    sbatch experiments/isaac/geo_dedup/verify_bruteforce/verify.sbatch

`verify.sbatch` first runs `tests/test_geo_dedup_verdict.py`, then `verify.py` from a NEW
directory `results/geo_dedup/verify_bruteforce_<job id>/` (it refuses an existing one). The
script prints one PASS / FAIL line per check (`verdict.py`: tolerances and their reasons), then
`GATE brute-force verify PASS|FAIL`, and exits 1 on FAIL. It refuses to overwrite `results.json`
or a sample csv. Success is the line `VERIFY_BRUTEFORCE_OK` in the log, not the job state.

## Executed result: job 6590222, 2026-11-12, seed 0

`GATE brute-force verify PASS: 9/9 checks` (log `logs/verify-bruteforce-6590222.out`, 3 min 12 s
on `short`; outputs in `results/geo_dedup/verify_bruteforce_6590222/`, not tracked). The verdict
tests passed in the same job (27 passed) before the check ran.

| check | n | fail | max discrepancy (tolerance) |
|---|---|---|---|
| city boxes rebuilt from `so2sat_cities.csv` | all | 0 | 1.421e-14 deg (1e-09) |
| v2 excluded rows have gap <= 1000 m | 2,000 | 0 | 0.198606 m vs matches min_distance_m (1.0) |
| v2 retained rows have gap > 1000 m | 20,000 of 8,039,068 | 0 | closest retained row 1000.89 m |
| v2 excluded is a subset of v3 excluded | 418,713 | 0 | 0 rows not excluded in v3 |
| v3-only excluded rows lie within 1000 m of a city box | 2,000 | 0 | 0.205716 m (1.0) |
| v3 retained rows outside every city buffer | 2,000 of 7,002,460 | 0 | nearest 1074.47 m |
| matches.csv distance_m vs independent gap | 500 | 0 | 0.104280 m; zero gaps 91 / 91 |
| tile bounds vs formula and meta parquet, name vs z/x/y | 20 | 0 | 0 deg |
| SSL4EO box centre vs authors' centre (n // 4 rule) | 20 | 0 | 0 m; half extent 5.4e-10 m |

The retained-row checks are SAMPLES (20,000 of 8,039,068 for v2; 2,000 of 7,002,460 for v3), so
they bound the rate of a wrong retention, they do not prove zero. "Closest retained row" is the
closest row WITHIN the sample (`c3_closest200.csv` = the 200 nearest of the 20,000 sampled), not
the closest of all retained rows.

The earlier run of 2026-10-13 was PRE-GATE: it predates `verdict.py`, printed no gate line, and
took its maxima with pandas (NaN-skipping; `verify.py` now uses numpy, which propagates a NaN into
a FAIL). Its numbers agree with the table above to the digits it recorded. Its `results.json`
next to this file is a local, untracked leftover; cite job 6590222, not that file.

The artifacts were produced before `box_gap_m` moved its shared latitude to the poleward end;
that version read up to 0.2 m high on a few pairs (never low), so no retained row can have a
true gap below 999.8 m. The per-row samples with the independent gap are written next to the
script when it runs (`c2_*.csv` ... `c5_*.csv`, not tracked).

## Ellipsoid check (`ellipsoid_check.py`, 2026-10-14)

`box_gap_m` uses a sphere (IUGG mean radius). Vincenty on WGS84 for a pair the rule reads as
exactly 1,000 m: true gap 994.4 m (north-south, equator; sphere over-reads, the leak side) to
1,004.1 m (east-west, 70 deg; sphere under-reads, over-exclusion). Max |error| 5.6 m. Output
of the script is the table; PROTOCOL.md section 1 quotes the bound.

