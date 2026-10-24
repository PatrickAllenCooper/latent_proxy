# Pinned tracer calibration v74: failed startup, identity gate passed

Job34837033 used1CPU/256MiB,7allocated seconds,0.198consumed CPU-seconds and57,628KiB peakRSS; zero GPUs. The frozen copied tracer passed the executable identity guard and preparation receipt. The capture immediately rejected one152-byte stderr line as unexpected child output, then stopped its owned child process. Zero trace bytes and no toy completion receipt survived. Static phase-code inventory is not evidence that either toy phase occurred.

This resolves the previous executable path/hash mismatch for this pinned copy. It does not establish tracer compatibility or permission, successful file tracing, or framework startup. The raw error was deliberately not saved, so shared-library incompatibility, program-prefix handling, and other startup errors remain hypotheses. Do not claim a specific cause from the line length or fast stop.

All six retained artifact hashes match remote copies; preparation and terminal metadata schemas pass. Runtime receipts total1,434bytes, within the unchanged64KiB aggregate budget. Cumulative diagnostic allocation is2,449CPU-seconds and254MIG-slice-seconds. No retry/new allocation/model work occurred. All unrelated jobs and remote artifacts were preserved.

The next proposal should distinguish bounded startup categories for the explicitly pinned tracer path without recording arbitrary stderr. A future launch must be separately admitted; this attempt is consumed. The narrow monitoring automation is removed after reporting this terminal outcome. CURC ticketINC1501930 remains informational and must not trigger a duplicate calibration.
