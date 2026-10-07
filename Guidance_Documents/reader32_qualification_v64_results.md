# Changed 32B qualification: incomplete at first response

Job 33507794 ended FAILED 2:0 after 90 allocated seconds on 6 October 2026 (18:00:05–18:01:35 America/Denver). One generation call was attempted; zero responses completed. Qualification is incomplete, with no accuracy estimate or scientific reader verdict. No retry was submitted.

The immutable source/sibling spool repair passed guarded imports. Model loading took 46.185947 seconds; Python start to model loaded took 69.942975 seconds, within the 90-second startup gate. All parameters passed CUDA BF16 assertions. Increasing CUDA allocation and the 771-weight loader independently establish actual GPU loading work. Peak framework allocated/reserved memory was 65,569,818,624 / 65,588,428,800 bytes (61.066 / 61.084 GiB), below the 65 GiB cap on a 69.75 GiB device. Host MaxRSS was 65,636,028 KiB (62.595 GiB), close to 64 GiB requested; retain host RAM.

The first frozen case exceeded its 10-second deadline. Stop receipt appeared after 10.044406 seconds, stop event after 10.045906; the 50 ms polling guard can overshoot. No partial token, stack trace or completed CUDA span survives because responses are saved only after generate returns. Generation-phase memory increases cannot establish throughput or distinguish cold setup from computation. NVML utilization was unavailable/unattributable. Internal token count is unknown.

All five artifact SHA-256 digests match remote copies in custody.json, including the empty response file. Source and frozen instrument are preserved. Requested/observed: one H200 3g.71gb, six CPUs, 64 GiB host RAM, 300-second requested wall, 90-second elapsed. TotalCPU 37.080 seconds; allocation costs 540 CPU seconds and 90 MIG slice-seconds. Retained stages total 1,717 allocated CPU seconds and 155 MIG slice-seconds (65 + 90); prior inconsistent CPU consumption snapshots remain preserved.

See reader32_first_generation_resource_proposal.md for the specific unsubmitted amendment. This result establishes memory feasibility, not preference learning, reader accuracy, or broader tool/prompt/fine-tuning effectiveness.
