# margin-referee

Can you ship this change? Ask your own eval set.

`margin-referee` tests a change on the cases you already have (a promptfoo results file, a
Braintrust, LangSmith or Langfuse dataset export, or JSONL): a new model, a new system prompt, or
the same model on another endpoint. It runs what you use today and the candidate, grades both with
each case's own check, and answers promote (ship it), a refusal (don't), or hold (not enough cases
to tell). What the change does to cost is one line of the answer. No account, no backend, no
telemetry.

Published on PyPI (https://pypi.org/project/margin-referee/) and GitHub
(https://github.com/trymargin/margin-referee).

```
pip install margin-referee
margin-referee fit cases.jsonl
OPENROUTER_API_KEY=... margin-referee replay cases.jsonl \
    --baseline openai/gpt-4.1 --candidate openai/gpt-6-luna --cap 1
```

## Why another eval tool

Because "the candidate scored higher on our evals" is how most bad switches start. Three things
make it unreliable, and this tool corrects for each:

- **Small sets lie.** The verdict is a paired test on the cases both models ran. Only the cases
  they disagree on carry evidence, and a clean sweep on 20 cases proves nothing. `fit` tells you,
  for free and before anything runs, how many cases you need: about 100 against a current model
  that passes everything, and more as its pass rate falls.
- **The best of several looks better than it is.** Try four models on the same cases and keep the
  winner, and you have mostly found noise. The bar rises with every candidate you try on
  overlapping cases (remembered in `~/.margin-referee/ledger.json`), so renaming or trimming the
  cases does not reset it. To confirm a winner, replay it alone on cases none of them saw.
- **Half-graded cases flatter the candidate.** A case with a check this tool cannot run (an LLM
  rubric, say) is left out and counted, never half-graded. An answer cut off by the output limit,
  or a call that errored, is unanswered, not a fail, and any unanswered case blocks a promotion.

## What `replay` also does

- **It stops paying once the answer is clear.** From 40 cases on, the replay stops as soon as the
  answer is already `do_not_ship`: even the reading most favourable to the candidate misses the
  bar. Each look is held to a stricter test (Bonferroni across the planned looks), so looking often
  cannot manufacture a loss. It never stops early for ship; `stopped_early` says when it stopped,
  and `--no-early-stop` runs every case. (Borrowed from VERDICT, `verdict-gate`.)
- **Reliability, with `--repeats 3`** (1 to 5). Each case runs that many times per side and passes
  only if every run passed (pass^k); the paired test reads that strict result. `reliability` gives
  pass@1 and pass^k per side, the cases that flipped between runs, and an estimate of the repeats
  needed to see a 1-point gap. Spend scales with the repeats.
- **Traffic weights.** Give a case a `weight` (or `metadata.weight`: its share of traffic, or a
  count) and `weighted` reports the same cases weighted by traffic, including the share of traffic
  on the cases that got worse. It is a weighting of the same cases, not a new test: the decision
  stays on the unweighted paired test.

## A second opinion on a comparison you already ran

Ran the baseline and a cheaper model as two Braintrust experiments (by hand, or with Loop)?
`check` reads both, pairs the cases, and gives the same verdict, without calling a model:

```
margin-referee check baseline.jsonl candidate.jsonl --scorer correct
BRAINTRUST_API_KEY=... margin-referee check bt:<baseline id> bt:<candidate id> --scorer correct
```

A pass is the named score at or above `--threshold` (default 1.0, full credit only). Pass
`--candidates N` when several models were tried on the same cases. A case one experiment lacks
blocks the promotion. With no `metrics.estimated_cost` in the exports, it says whether quality
held but claims no saving. Exit code 0 means promote.

## The verdict on a Hurdle suite

Ran a [Hurdle](https://github.com/togethercomputer/hurdle) suite with and without a skill, or on
two models? `hurdle` reads the runs it saved and says whether the difference is real:

```
margin-referee hurdle .hurdle --compare condition --baseline baseline --candidate with-skill
margin-referee hurdle .hurdle --compare model --baseline <model now> --candidate <model to try>
```

A task passes only if every repeat passed, and `repeats_passed` shows each side's count. A run
Hurdle marked incomplete or unknown leaves its task unanswered, which blocks a promotion.

## A receipt for the verdict

`replay`, `check` and `hurdle` take `--receipt out.json`. It writes the verdict as an
[in-toto Statement](https://github.com/in-toto/attestation) in a DSSE envelope: the decision, the
bar, the paired counts and gap, how many candidates were tried, cost per case, each segment, and a
list of what the record does not show. A receipt you write this way is **unsigned** and says
`assuranceLevel: self-run`. It is your own claim in a format someone else can read.

A receipt Margin's hosted service issues is signed with Ed25519. Check one with:

```
pip install 'margin-referee[verify]'
margin-referee verify receipt.json --key-url https://trymargin.io/.well-known/margin-receipt-key
```

Exit 0 means the holder of that key signed it and nobody edited it since. It is a signed record,
not a certification: it does not show the grader is right, that the cases cover production, or
what a person decided to ship.

## What it runs on

- **OpenRouter**, with your own key: any model it serves.
- **A model in your own account**, with `--candidate-endpoint` or `--baseline-endpoint` and a
  JSON object: `bedrock` (region, model, a Bedrock API key), `vertex_ai` (project, region, model,
  access token), `azure_openai` (resource, deployment, key) or `openai_compatible` (base URL,
  model, key: a fine-tune, vLLM, a self-hosted model). Your credential is used for that run only,
  and you give your own per-token price, so a negotiated discount counts.
- **A prompt change**: `--candidate-system` replays the candidate with a new system prompt. Each
  prompt you try counts as a candidate.

## What it does not do

It does not open the pull request, watch the change after you merge it, keep the history across
your team, or let your agents call it with a spend limit. That is
[Margin](https://trymargin.io), which runs this same test.

MIT licensed.
