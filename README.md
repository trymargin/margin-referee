# margin-referee

Should you switch models? Ask your own eval set.

`margin-referee` replays the cases you already have (a promptfoo results file, a Braintrust,
LangSmith or Langfuse dataset export, or JSONL) on the model you run today and on a candidate,
grades both with each case's own check, and tells you whether the candidate holds up and costs
less. No account, no backend, no telemetry.

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
