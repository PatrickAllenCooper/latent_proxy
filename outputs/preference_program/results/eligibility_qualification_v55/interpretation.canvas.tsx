import { Stack, Row, H1, H2, Text, Table, Card, CardBody, Divider, useHostTheme } from "cursor/canvas";

const cases = [
  ["00", "A", "A", "Correct"], ["01", "B", "B", "Correct"],
  ["02", "C", "C", "Correct"], ["03", "D", "D", "Correct"],
  ["04", "A", "A", "Correct"], ["05", "B", "B", "Correct"],
  ["06", "C", "A", "Eligible; priority rank 3 instead of 2"],
  ["07", "D", "B", "Ineligible highest-priority option"],
  ["08", "A", "A", "Correct"], ["09", "B", "B", "Correct"],
  ["10", "C", "C", "Correct"],
  ["11", "D", "B", "Eligible; priority rank 2 instead of 1"],
  ["12", "A", "A", "Correct"], ["13", "B", "B", "Correct"],
  ["14", "C", "B", "Eligible; priority rank 4 instead of 2"],
  ["15", "D", "B", "Eligible; priority rank 4 instead of 2"]
];

export default function QualificationReport() {
  const theme = useHostTheme();
  return <Stack gap={20} style={{ padding: 24, maxWidth: 1050, color: theme.text.primary }}>
    <Text>Frozen no-advice qualification · Qwen2.5-1.5B-Instruct · 5 October 2026</Text>
    <H1>11 of 16 correct — qualification failed</H1>
    <Text>The registered threshold remains 16/16. All responses parsed and ended with EOS; the 56-response advice panel stays blocked.</Text>
    <Row gap={24} wrap>
      <Stack gap={4}><H2>4 priority errors</H2><Text>Feasible option, wrong ordering</Text></Stack>
      <Stack gap={4}><H2>1 eligibility error</H2><Text>Ineligible option selected</Text></Stack>
      <Stack gap={4}><H2>0 parse failures</H2><Text>No truncation; 16 EOS records</Text></Stack>
    </Row>
    <Divider />
    <H2>Every frozen case</H2>
    <Table headers={["Qualification case", "Gold", "Response", "Outcome"]} rows={cases} striped />
    <Text>Source: job 33469187, smoke/base.jsonl; independently recomputed gold from frozen qualification_cases.jsonl. All sixteen cases retained.</Text>
    <H2>Response counts by gold letter</H2>
    <Table headers={["Gold (4 cases each)", "Response A", "Response B", "Response C", "Response D"]} rows={[["A",4,0,0,0],["B",0,4,0,0],["C",1,1,2,0],["D",0,3,0,1]]} />
    <Text>Descriptive counts only. Gold letters cycle with run order; label, position and semantic differences are confounded. This is not a causal bias estimate or a population sample.</Text>
    <Card><CardBody><Stack gap={8}>
      <H2>Bounded, verified GPU execution</H2>
      <Text>One H200 35 GB MIG slice · 4 actual CPUs · 32 GB RAM · 34 seconds · peak CUDA reserved memory 2.93 GiB.</Text>
      <Text>32 generated tokens and nonzero CUDA event spans support real device work. MIG utilization was N/A; host-wide telemetry does not identify this job's memory. All artifacts preserved; commit 96a9cbb.</Text>
    </Stack></CardBody></Card>
    <H2>Next comparison: representation and relabeling</H2>
    <Text>Design only: two fresh semantic cases × four letter rotations × prose versus explicit priority ranks = sixteen responses. Freeze paired cases and randomized order first. This would be a mechanism smoke, not a rescue of the failed baseline.</Text>
    <Text style={{ color: theme.text.secondary }}>No new inference, GPU allocation, scorer changes, case changes or pending approval request. This report is a post-outcome interpretation.</Text>
  </Stack>;
}
