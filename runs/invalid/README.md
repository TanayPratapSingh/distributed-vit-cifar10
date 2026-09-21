# Discarded measurements

These gloo runs were measured while the MPS laptop suite was using the
same 10 cores. They reported 146, 288 and 343 images per second against
521, 632 and 814 from the idle machine, and the contention even reversed
the efficiency trend.

They are kept rather than deleted because the discrepancy is the point:
throughput measured under unquantified background load is not a
measurement. `dvit.report` only reads `runs/*.json`, so nothing here
reaches the dashboard.

Re-measure on an idle machine with `make gloo`.
