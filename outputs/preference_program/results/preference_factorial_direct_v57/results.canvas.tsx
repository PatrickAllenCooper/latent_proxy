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
    <H2>Finite next scientific gate</H2>
    <Text>Stop retuning this assay. Recommend an independently registered fresh capability comparison with one preselected stronger checkpoint and a deterministic constrained proxy control. Establish rule execution before advice or optimizer/tool/adapter comparisons.</Text>
    <Text style={{color:theme.text.secondary}}>No retry, replacement, new job, selected-case rescue or advice reopening. Rank annotation changed decisions but did not improve paired correctness.</Text>
  </Stack>;
}
