import { Stack, Row, Grid, H1, H2, Text, Table, Divider, useHostTheme } from "cursor/canvas";

export default function FactorialDesign() {
  const theme = useHostTheme();
  return <Stack gap={20} style={{ padding:24, maxWidth:1100, color:theme.text.primary }}>
    <Text>Frozen local design v56 · no inference submitted</Text>
    <H1>Separate preference representation, letter identity and row order</H1>
    <Text>Sixteen responses proposed. The existing 11/16 qualification remains failed and advice remains blocked.</Text>
    <Grid columns={3} gap={20}>
      <Stack gap={8}><H2>1. Two fresh semantic cases</H2><Text>One top-priority option eligible; one ineligible. Exclude all 24 prior semantic signatures, regardless of labels or row order.</Text></Stack>
      <Stack gap={8}><H2>2. Eight configurations per case</H2><Text>2 representations × 2 letter assignments × 2 catalog orders. All three factors independently crossed.</Text></Stack>
      <Stack gap={8}><H2>3. Sixteen paired responses</H2><Text>Fixed seeded run order. Every gold letter and best-option row position appears four times. No favorable subset selection.</Text></Stack>
    </Grid>
    <Divider />
    <H2>Configuration matrix — repeated for each semantic case</H2>
    <Table headers={["Representation", "Letter assignment", "Catalog order"]} rows={[
      ["Prose", "Canonical", "Forward"], ["Prose", "Canonical", "Reverse"],
      ["Prose", "A↔C / B↔D", "Forward"], ["Prose", "A↔C / B↔D", "Reverse"],
      ["Explicit ranks", "Canonical", "Forward"], ["Explicit ranks", "Canonical", "Reverse"],
      ["Explicit ranks", "A↔C / B↔D", "Forward"], ["Explicit ranks", "A↔C / B↔D", "Reverse"]
    ]} />
    <Text>Source: preference_factorial_v56/execution_manifest.jsonl; design seed 2026100601. This matrix specifies configurations, not measured outcomes.</Text>
    <H2>Paired scientific contrasts</H2>
    <Table headers={["Eight pairs per factor", "Held fixed", "Resolved within these cases"]} rows={[
      ["Rank annotation", "Content, letters, catalog order", "Sensitivity to externally computed preference ranks"],
      ["Letter rotation", "Content, ranks, catalog order", "Output-letter identity sensitivity"],
      ["Catalog reversal", "Content, ranks, letters", "Catalog-position sensitivity"]
    ]} />
    <Text>Content is a two-case blocking variable. Content effects and eligibility difficulty are not separately identified; more independent semantic blocks require confirmation.</Text>
    <H2>Finite next gate</H2>
    <Text>Require complete provenance, token equality, normal termination and all sixteen correct. If ranks succeed while prose fails, confirm structured proxy representation on new cases. If letter/order invariance fails, retain the sensitivity diagnosis. If both formats fail, compare a preselected stronger checkpoint or deterministic proxy control.</Text>
    <Row gap={24} wrap><Text>Proposed ceiling: one H200 35 GB MIG · 4 requested CPUs · 32 GB RAM · 5 minutes</Text><Text>CPU preparation ≤4 actual CPUs / 8 GB / 90 seconds</Text></Row>
    <Text style={{color:theme.text.secondary}}>Design and synthetic fixtures only. No new approval request. No new model calls, GPU jobs, scorer changes, threshold relaxation or advice authorization.</Text>
  </Stack>;
}
