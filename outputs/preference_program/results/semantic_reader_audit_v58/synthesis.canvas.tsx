import { Stack, H1, H2, Text, Table, Divider, useHostTheme } from "cursor/canvas";
export default function FactorialResults() {
  const theme = useHostTheme();
  return <Stack gap={20} style={{padding:24,maxWidth:1100,color:theme.text.primary}}>
    <Text>Frozen v56 factorial · job 33483615 · 6 October 2026</Text>
    <H1>6/16 correct — representation smoke failed</H1>
    <Text>Valid execution: all sixteen responses, zero parse failures, sixteen EOS, exact CPU/GPU tokens and provenance. The advice panel stays blocked; v55's failed baseline remains intact.</Text>
    <Table headers={["Format (8 cells each)","Correct","Ineligible choices","Eligible priority errors"]} rows={[["Prose",3,2,3],["Explicit ranks",3,4,1]]} />
    <H2>All frozen factor contrasts</H2>
    <Table headers={["Paired manipulation","Mean accuracy difference (percentage points)","Same semantic choice (of 8 pairs)","Correctness changes"]} rows={[
      ["Prose → explicit ranks",0,4,"All eight differences zero"],
      ["Canonical → rotated letters",-25,5,"Two losses; six ties"],
      ["Forward → reversed catalog",25,5,"Two gains; six ties"]
    ]} />
    <Text>Source: frozen factorial_analysis.json. Differences are second minus first within exact paired cases. These are two constructed semantic blocks; no population confidence interval or causal generalization.</Text>
    <H2>Semantic case outcomes</H2>
    <Table headers={["Fresh semantic case","Prose","Explicit ranks","Diagnostic"]} rows={[
      ["fresh-00: top priority eligible","3/4","3/4","Rotated forward labels fail; reversal restores correctness"],
      ["fresh-01: top priority ineligible","0/4","0/4","Rank format chooses ineligible top option in every cell"]
    ]} />
    <Divider />
    <H2>Verified bounded resources</H2>
    <Text>CPU gate 33483501: four CPUs, 8 GB, 56 seconds; reference and fresh token IDs matched. GPU: one H200 35 GB MIG, four CPUs, 32 GB, 28 seconds; peak reserved CUDA memory 2.93 GiB.</Text>
    <Text>CUDA memory, nonzero device event spans and 32 generated tokens verify work. MIG utilization was N/A; host-wide telemetry cannot be attributed to this job. Prior timeout and negative outcomes are preserved.</Text>
    <H2>Independent existing-output audit</H2>
    <Text>All 32 v55/v56 prompts independently parsed, rendered, re-encoded and scored. Gold, eligibility, ranks, labels, row transformations and EOS agree with the saved results. The preserved local tokenizer is not byte-identical to the pinned production JSON; this is sequence consistency alongside production hash receipts.</Text>
    <Text>A deterministic constrained reward policy matches an independent priority-walk oracle on all 2,304 finite assignments. This validates computational execution, not LLM tool use or preference discovery. The two panels are not pooled as independent user observations.</Text>
    <H2>Recommended next decision</H2>
    <Text>Use the deterministic constrained proxy as the execution control and close the current 1.5B reader instrument as a negative result. A single preselected 3B checkpoint on sixteen fresh semantic cases is a conditional proposal only if internal LLM execution remains necessary. No prompt search or rerun of these assays.</Text>
    <Text style={{color:theme.text.secondary}}>No retry, replacement, new job, selected-case rescue or advice reopening. Rank annotation changed decisions but did not improve paired correctness.</Text>
  </Stack>;
}
