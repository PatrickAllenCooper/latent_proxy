# Approved attempt: preflight path alias hold

The one-attempt first-case 10→30 second amendment is approved and separately bound in commit 9602b83. The original e027985 review freeze remains unchanged. No GPU job was submitted, no model loaded or response generated, and no approved attempt consumed.

Fresh source-bundle, runtime archive/receipt, Python and authorization checks passed before the cached-snapshot guard stopped. All 24 saved model/tokenizer file records retain their sizes and nanosecond mtimes. Each mismatch is solely the resolved path prefix: saved `/gpfs/alpine1/scratch/...` versus current `/scratch/alpine/...`. On login-ci4, os.path.samefile confirms every saved/current pair names the same file. This is demonstrated aliasing, not evidence of changed weights. The full saved weight digests are preserved; no new bulk weight hash was performed on the login node.

The approved frozen runner still checks literal resolved-path equality. Its behavior in the current compute-node mount namespace has not been checked; allocating a GPU just to discover that startup mismatch would be inefficient. As instructed for preflight ambiguity, stop before submission. Preserve staged immutable source, spool diagnostics and both prior failed attempts (65 +90 MIG seconds); no retry or guard relaxation occurred.

Smallest next dependency: a CPU-only check of exact path resolution and same-file identity in the compute-node context, followed by a separately reviewed engineering binding only if required. Scientific inputs, sixteen cases, strict scoring, 24-token limit, model settings and approved 90/30/10/270/285/300 limits remain intact. No further numerical cap approval is needed; the existing approved GPU attempt remains available after this readiness issue is closed.
