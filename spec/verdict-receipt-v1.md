# Verdict receipt v1

A verdict receipt records one decision about one AI change: was the candidate setup (a model,
prompt, tool or host) as good as the baseline on the same cases, and at what cost. Anyone can
check it without trusting the tool that produced it.

## Format

- **Envelope:** DSSE (`payloadType: application/vnd.in-toto+json`), with the signature computed over
  the DSSE pre-authentication encoding `DSSEv1 <len(type)> <type> <len(payload)> <payload>`.
- **Payload:** an in-toto Statement v1 with `predicateType: https://trymargin.io/verdict-receipt/v1`.
  The JSON Schema for the statement is `verdict-receipt-v1.schema.json` in this directory.
- **Subjects:** the sha256 of each side's setup string, named `baseline` and `candidate`.
- **Signature:** Ed25519. A receipt Margin issues names its key by `keyid`, the sha256 of the raw
  public key. The public key is served at `https://trymargin.io/.well-known/margin-receipt-key`.
- **Unsigned receipts** have an empty `signatures` list. `margin-referee replay|check|hurdle|ontology-sql
  --receipt out.json` writes one with `assuranceLevel: self-run`. It is your own record, and its
  format is the same.

## Predicate fields

| field | meaning |
|---|---|
| `decision` | `ship`, `ship_if_worth_the_cost`, `do_not_ship` or `not_enough_evidence` |
| `why` | the reason in one sentence |
| `decisionRule` | the metric, the bar (0.97 of the baseline's pass rate) and how the bar rises with each candidate tried |
| `paired` | cases compared on both sides, cases won and lost, pass counts, and the quality gap (estimate, worst case at the gate's confidence, and what the bar allows) |
| `candidatesTried` | how many setups were tried on these cases; the bar is corrected for it |
| `costPerCase` | each side's mean cost per case, or null when no cost was logged |
| `segments` | per labelled segment: cases, lost, won, and whether it is proven worse or, for a release-critical segment, proven to hold |
| `worseCases` | the first cases the candidate got wrong that passed before |
| `casesDigest` | a digest of the case set when known, else null |
| `assuranceLevel` | `run-by-margin` (Margin replayed the cases), `scored-by-customer` (Margin judged two experiments the team ran), `self-run`, or `unknown` |
| `nonClaims` | what the receipt does not establish, inside the signed bytes |

## What a receipt is not

A valid signature shows who issued the receipt and that nobody edited it since. It does not show
that the grader was fit for purpose, that the cases cover production, or that a person decided to
ship. It is not a certification or an attestation of compliance. A team's reported outcome after
shipping is kept out of the receipt on purpose: a receipt is never edited.

## Verify

    pip install "margin-referee[verify]"
    margin-referee verify receipt.json --key-url https://trymargin.io/.well-known/margin-receipt-key

This spec is open. Anyone may write or check receipts in this format.
