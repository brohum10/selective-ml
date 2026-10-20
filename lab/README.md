# Local inspector assets

These buildless assets are served only by `lab_server.py` on loopback. They make
real requests to the local experiment worker; no scores or outcomes are fabricated
in the UI and no hosted API or third-party assets are used.

The decision workbench adds a filterable error atlas, full confusion matrix,
fixed-bin reliability table, and threshold sandbox. Counts and exact attainable
threshold points come from `diagnostics.py`, which is tested against the recorded
predictions and brute-force counts. The browser uses binary search on that curve.
The sandbox changes only the exploratory view, never the frozen policy or gate.
Test-set exploration must not be presented as a fitted or validated policy.

From the repository root:

```sh
python lab_server.py --evaldelta ../evaldelta/evaldelta.py
```

Open http://127.0.0.1:8766 and run the experiment. The UI polls actual worker events,
reports every planned seed, and lets you inspect gates, review decisions, raw JSON,
and the completed evidence ZIP. Unknown filesystem paths are never served.

The server is not production hosting. No identity system, multi-tenant isolation,
automatic deployment or human-review queue is implemented. CSRF/Origin/Host checks
and bounded execution protect the local demonstration; do not expose it publicly.
