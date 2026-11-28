# margin-referee

Can you ship this change? Ask your own eval set.

`margin-referee` tests a change on the cases you already have (a promptfoo results file, a
Braintrust, LangSmith or Langfuse dataset export, or JSONL): a new model, a new system prompt, a
changed tool list, or the same model on another endpoint. It runs what you use today and the
candidate, grades both with each case's own check, and answers promote (ship it), a refusal (don't),
or hold (not enough cases to tell). What the change does to cost is one line of the answer. No
account, no backend, no telemetry.

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
  It does not hide a loss: if the change is still worse than the bar allows with every unanswered
  case counted as a win for the candidate, the answer is `do_not_ship`, not `not_enough_evidence`.

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

## Runs that cannot reach a verdict are refused before they spend

A paid replay that ends `not_enough_evidence` for a reason you could have known first is wasted
money. `replay` checks for those reasons in order and stops at the first one it finds. The
decision is then `refused_before_spend`, the reason is the first line printed, and the exit
code is 3.

1. **Size, free.** If you have fewer gradable cases than the gate needs (`cases_needed`, 100
   for one candidate), no result could be `ship`, so nothing runs. `--anyway` runs it as a
   check that can only find a loss, and the output says so.
2. **Smoke, a few cents.** The first two cases run on both models. If a call errors (a model
   id that does not exist, a 4xx or 5xx, a schema the endpoint rejects, an answer the check
   cannot read), the run stops with the exact error and what it spent. An answer cut off by
   the output limit is retried once at double the limit, up to 32768 tokens. If it is still
   cut off, the run stops and names the limit. Set the limit with `--max-tokens`.
3. **Budget.** The full run is estimated from the two smoke cases plus 20%. If that is more
   than `--cap`, the run stops and names the cap it would need. On OpenRouter it also reads
   what your key can still spend. If the estimate is more than that, the run stops. If the
   balance cannot be read, the output says `balance not checked` and the run goes on.

The two smoke cases are the first two cases of the full run, so a run that goes ahead pays
for nothing twice.

Three more guards work during the run:

- A 402 (out of credit) stops the run at once. Every later call would fail the same way.
- A case lost to a 429, a 5xx, a timeout or a dropped connection is tried once more when the
  run ends. So is an answer cut off by the output limit, at double the limit. A 400, 401, 403,
  404 or 422 is not retried: the same call would fail again.
- If so many cases have no answer that the rest cannot reach `cases_needed`, the run stops
  and says so. No later case could have made the verdict `ship`.

Every verdict carries a `preflight` block: `size`, `smoke`, `budget`, `estimate_usd`,
`balance_usd` and the `max_tokens` the run ended up using.

Exit codes for `replay`: 0 promote, 1 do not promote, 2 no case can be graded, 3 refused
before spend.

## Testing a tool change

A tool description is a prompt the model reads on every call. Rewrite one, or add or remove a
tool, and the agent may start calling the wrong tool. To test that before it ships, give each
JSONL case the tools the call offers and say which tool it should call:

```
{"id": "refund-policy", "input": "How long do I have to return shoes?",
 "tools": [{"type": "function", "function": {"name": "search_kb", "description": "Search.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}],
 "check": "tool-call", "expected": "search_kb"}
```

- `tools` is an OpenAI-style function tool list, and `tool_choice` is optional. Both are sent
  with every call, as they are.
- `check: "tool-call"` passes when the model's first tool call is the `expected` tool. To check
  arguments too, write pairs separated by `;`, such as
  `name=lookup_order;args.order_id=A1042`. Values compare as text, ignoring case.
- No tool call is a fail. A call whose arguments cannot be read, or an endpoint that does not
  report tool calls, leaves the case unanswered. It is never counted as a pass.

Then put the new tool list in a file and replay with `--candidate-tools`. The candidate gets
that list in place of each case's tools; the baseline keeps the case's own. The model can stay
the same:

```
cd examples/tool_change
OPENROUTER_API_KEY=... margin-referee replay cases.jsonl \
    --baseline openai/gpt-4.1 --candidate openai/gpt-4.1 \
    --candidate-tools candidate_tools.json --cap 1 --anyway
```

`examples/tool_change` has three cases and a `candidate_tools.json` that rewrites two vague
descriptions. Three cases show the format and prove nothing: without `--anyway` the replay is
refused before it spends, with it the verdict is `not_enough_evidence`, and `fit` says how many
cases you need. Each tool list you try counts as
a candidate, the same as a prompt. Tool cases run on OpenRouter and on OpenAI-compatible, Azure
OpenAI and Vertex AI endpoints; a Bedrock endpoint leaves them unanswered for now. A promptfoo
results file carries no tool list, so write tool cases as JSONL.

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

## Judging an ontology layer: NVIDIA ontology-sql-eval

Ran [ontology-sql-eval](https://github.com/NVIDIA/ontology-sql-eval) to compare a schema-only
prompt with the ontology-grounded agent? `ontology-sql` reads the two arms' result files and
says whether to ship the layer. It calls no model and needs nothing of NVIDIA's installed:

```
margin-referee ontology-sql runs
margin-referee ontology-sql runs/full_20261126 --price-in 5 --price-out 25 --price-cached 0.5 --json
```

It reads `<run>/control/<dataset>.jsonl` and `<run>/treatment/<dataset>.jsonl`, pairs the
questions by `task_id`, and applies the referee's rules, which are stricter than the harness's
own report:

- A pass is the verifier's positive verdict. A wrong result, a query that errored or timed out,
  or no SQL in the answer is a fail.
- A gold query that did not run, a run with no model output, an unknown error, or a question
  one arm has no row for is unanswered. It is listed with its reason, counts as neither a pass
  nor a fail, and blocks a promotion. The harness scores some of these as misses, so its
  accuracy and this one can differ on the same run.
- Tokens per question come from the per-call capture block on each row when the run recorded
  one, otherwise from the ontology arm's own totals. Cost is those tokens times the
  `--price-in` and `--price-out` you give (USD per million tokens). The files carry no price,
  so without both flags cost reads `not_recorded` and nothing is guessed.
- Cached input is read per call (`cached_tokens` in the capture block, otherwise the ontology
  arm's `agent_cached_input_tokens`) and is part of the input count. Give `--price-cached`
  (USD per million cached input tokens) to price it at its own rate. Without that flag cached
  input is priced as fresh, and the cost basis says so. The cached rate is used only when both
  arms recorded the split; if either did not, the output says `cache split not recorded` and
  all input is priced as fresh. No split is guessed.
- The cache-hit share (cached input over all input) is printed for each arm, and the JSON has
  `cache: {baseline: {fresh_in, cached_in, share}, candidate: {...}, line}`, totals over the
  paired questions. When the two arms differ by ten points or more you get a line such as
  `the candidate's cache-hit share fell from 80% to 12%`, so a change that breaks prompt
  caching shows up. The share changes cost only. It never changes a pass, a fail or the
  quality verdict.
- The ontology arm makes many more calls per question. If the pass rate only holds, the answer
  is `not_enough_evidence`: a layer that is not cheaper ships on a proven gain. If the gain is
  proven it reads `ship_if_worth_the_cost`, with the questions won and lost and the 95% bound
  on the gain, and the decision to pay is yours.
- Results are also read per `difficulty` (or `db_id`; choose with `--segment-by`), so a gain
  overall cannot hide a loss on the hard questions.

`--baseline-arm` and `--candidate-arm` name the two folders, or two rollout files, if yours are
called something else. Swap them to ask the opposite question: can the layer be dropped?

The file layout and field names were read from the harness source at commit `871bb924`
(2026-11-26) and the tests use rows made in that shape. The cached-input field names were read
from one real run's files (WideWorldImporters, 41 questions, 2026-11-27). Prices you pass are
yours: a list price is not what a gateway billed.

## A receipt for the verdict

`replay`, `check`, `hurdle` and `ontology-sql` take `--receipt out.json`. It writes the verdict as an
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
- **A tool change**: `--candidate-tools` replays the candidate with a new tool list (see
  "Testing a tool change").

## What it does not do

It does not open the pull request, watch the change after you merge it, keep the history across
your team, or let your agents call it with a spend limit. That is
[Margin](https://trymargin.io), which runs this same test.

MIT licensed.
