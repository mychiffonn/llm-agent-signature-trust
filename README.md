# Signed isn't independent

**Verified is not the same as corroborated.** A signature tells an agent *who* signed a report. It does not tell the
agent whether the report is true, or whether five signed reports come from five independent witnesses or from one
operator with five keys.

Project for the Apart AI x Epistemics Research Sprint (13-15 Nov 2026), **Track 2: Trust Infrastructure**. The track's
example idea is "a controlled study of whether provenance signals help AI research agents". We run that study under
attack.

Results page: `summary.html` (rebuild with `python -m harness.report`). Every number on it comes from `results/`.

## What we found

- **Agents over-weight signatures.** In three of four models, a signed report moves belief more than the same report
  unsigned. When only the minority side is signed, Gemini 3.8 Flash sides with it and calls it "verified".
- **Attackers can use that.** Fake witnesses that sign gain influence while honest witnesses do not. A forged signature
  is caught and ignored.
- **An independence count removes the fakes, and honest sources with them.** A tool that counts distinct accredited
  witnesses removes the fakes' influence in three of four models. It also discards honest witnesses who do not sign or
  are missing from the registry, and it cannot detect a captured witness.
- **The effects spread in a network** of 15 agents passing reports for 5 rounds.

**For builders of agent platforms:** show agents the number of independent accredited sources behind a claim, not a
"verified" badge. State what a signature does and does not prove. Measure which honest sources your checks remove.

## Why this matters

Agents increasingly form beliefs from what other agents and pipelines pass on: research agents summarising field
reports, agent-to-agent protocols, retrieval over sources that carry content credentials. The cheapest way to mislead
such an aggregator is a **manufactured consensus**: many voices that are really one. Provenance comes in three layers
that do different jobs:

| Layer | What it proves | Stops fake witnesses? | Good for |
|---|---|---|---|
| Signature | This exact text was signed by this key and not altered | No: a fake makes its own key for free | Catching forged attributions and tampering |
| Registry | The key belongs to an accredited first-hand witness | Only if the registry is complete and honest | Turning a key into an identity |
| Independence count | How many distinct accredited origins stand behind a claim | Yes, within the registry's assumptions | The only layer that addresses manufactured consensus |

Adoption makes this sharper. An attacker can sign on day one at no cost, while honest sources sign late or never.

## The study

Each case poses a question with two answers. Five accredited witnesses each hold one report: three point to the truth
and two are honest mistakes. An attack adds fake reports. The agent reads the same inbox in every condition, and only
who signs and what the agent is shown change. The agent answers with a probability for each answer.

| | Question | What changes (same reports) | Prediction |
|---|---|---|---|
| **E1** | Do agents over-read signatures? | Who signs: the true side, the false side, everyone, no one | **E1a** a signed report outweighs the same report unsigned. **E1b** some of that extra weight survives when every attribution is guaranteed accurate. **E1c** explaining what a signature proves shrinks it. **E1d** a signature on every record changes nothing. |
| **E2** | Can an attacker buy that weight? | Only the fakes sign vs no one signs; a report forged in a real witness's name | **E2a** a fake pulls harder when it signs and honest sources don't. **E2b** a forged report moves signed records less than unsigned ones. |
| **E3** | What does an independence count buy and cost? | Count on/off; registry coverage 100% / 60%; honest sources unsigned; a captured witness | **E3a** it removes fakes. **E3b** an incomplete registry costs honest evidence. **E3c** it discards honest sources that don't sign. **E3d** a captured witness passes every check. |
| **E4** | Does it compound in a network? | 15 agents, 5 rounds; only fakes sign vs everyone signs | **E4a** fakes that sign while honest agents don't do more harm. **E4b** with the count, honest unsigned evidence is discarded. |

Predictions were written after an exploratory pilot and tested on fresh cases (3 per condition), with Student-t
intervals. The page shows pilot and confirmatory results separately.

### What the agent sees

| Condition | What the agent sees and can do |
|---|---|
| Plain text | Each report as a chat message: "Passing on a first-hand report from Lena Fischer: ..." |
| Unsigned records | The same message plus a record {claim, origin, relay chain} with no signature field |
| Signed records | The record also shows the platform's verification result: "valid: signed with Lena Fischer's key", "none: this record is not signed", or "INVALID: signature does not match" |
| Signed, attribution guaranteed | Signed records plus: every origin is accurately attributed, signed or not. This removes the one good reason to discount an unsigned record. |
| Signed, explained | Signed records plus: a valid signature shows who signed, not that the origin is accredited; repeated records from one origin count once |
| Independence count | The explanation plus two tools: `verify_record(id)` (signature valid? origin registered?) and `count_independent_origins()` (distinct origins per answer that are validly signed **and** registered) |

### Attacks

- **Fake witnesses:** one or three new identities with their own keys report the wrong answer.
- **Echo:** one fake report passed on by three different people.
- **Forgery:** a wrong report attributed to a real witness who sent nothing; its signature does not verify.
- **Captured witness:** a right witness signs a false report with its real, registered key.

Fakes use the same phrasings as real reports, so only provenance can give them away.

### Measures

- **Report-equivalents:** log-odds of the agent's belief in the truth divided by log 3, clipped to 1-99%. One genuine
  report moves a Bayesian by exactly 1.
- **Signature premium:** (belief with the true side signed minus belief with the false side signed) divided by the
  number of reports. This is the extra weight of one signed report over the same report unsigned.
- **Brier Index:** 100 x (1 - Brier / 0.25). 100 is certain and right, 0 is a coin flip, below 0 is confidently wrong.
- **Benchmarks:** a naive counter (each distinct claimed origin counts once) and a registry checker (only accredited
  origins count).

### Scope

Every report is equally strong, so the task reduces to deciding which origins to count. That is the decision
provenance is supposed to inform, which makes its effects identifiable. It says nothing about judging content. The
adversary module is a recipe for laundering a false claim through signatures; everything is synthetic and the attacks
are not tuned for strength.

## Positioning

| Prior work | What it did | What we add |
|---|---|---|
| Gao et al. 2026, *Copies or Sources?* (arXiv 2610.06192) | LLM aggregators over-count restated evidence. Provenance was honest. | Adversarial provenance: fakes with valid signatures, captured witnesses. |
| Yan et al. 2026, *When Truth Is Distributed* (arXiv 2608.03421) | One deceiver derails 5-agent groups; they recommend provenance and independence checks but do not test them. | We test them and find which part does the work. |
| Xie et al. 2026, *From Spark to Fire* (arXiv 2603.04474) | An external lineage plugin contains error cascades. | Provenance that agents consume themselves, and the cost of an incomplete registry. |
| Prakash 2026, *LDP* (arXiv 2603.08852) | Noisy provenance metadata hurt synthesis quality. | Unsigned vs signed vs counted, with transmission held fixed. |

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install anthropic networkx cryptography pytest httpx numpy scipy matplotlib
.venv/bin/python -m pytest -q tests                                   # provenance unit test
.venv/bin/python -m harness.packet --mock --seeds 3                   # offline packet run (mock numbers are NOT results)
.venv/bin/python -m harness.run_b --mock --seeds 2 --topologies small_world   # offline network run

# E1 + E2 (one model; repeat per model with --backend/--model)
X="--badge --realism natural --reliability stated unstated --seeds 3 --seed0 2000"
.venv/bin/python -m harness.packet --backend gemini --formats P2 P2R --packets none sock1 sock3 forged \
    --signing all none truth false puppets $X --out results/packet_conf_<model>_sig.jsonl
.venv/bin/python -m harness.packet --backend gemini --formats P2 --packets none sock1 sock3 forged \
    --signing all none truth false puppets $X --attribution-note --out results/packet_conf_<model>_attr.jsonl
# E3
.venv/bin/python -m harness.packet --backend gemini --formats P0 P2U P2 P3 --packets none sock3 echo3 captured forged \
    --coverage 1.0 0.6 $X --out results/packet_conf_<model>_checks.jsonl
.venv/bin/python -m harness.packet --backend gemini --formats P3 --packets none sock3 forged --signing puppets none $X \
    --out results/packet_conf_<model>_adopt.jsonl
# E4
.venv/bin/python -m harness.run_b --backend gemini --experiments P2 P3 --adversaries none sockpuppet \
    --signing all puppets --badge --seeds 3 --seed0 2000 --topologies small_world --out results/net_sig_<model>.jsonl
.venv/bin/python -m harness.report                                   # rebuild summary.html
```

Format codes: `P0` plain text, `P2U` unsigned records, `P2` signed records, `P2R` signed and explained, `P3`
independence count. Backends: `gemini`, `deepseek`, `openrouter`, `hf`, `api`. API keys are read from the environment
or a local `.env` (`GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `HF_TOKEN`, `ANTHROPIC_API_KEY`). Every model call is cached in
`cache.sqlite`, so rerunning a command fills only the gaps.

## Files

```
harness/scenarios.py   cases, evidence phrasings, names, reliability note, Bayes posterior
harness/provenance.py  keys, records (signed or unsigned view), originate/relay/forge/verify, origin counter
harness/packet.py      single-agent evidence packets (E1-E3): packets, signing patterns, badges, runner
harness/network.py     network (E4): graphs, prompts, adversaries, partial signing, simulation, exposure
harness/llm.py         model backends, JSON output, SQLite cache
harness/mock.py        offline heuristic agent for pipeline tests
harness/metrics.py     per-run metrics (Brier Index, benchmarks, exposure, cite and check rates)
harness/run_b.py       network sweep runner
harness/demo.py        readable trace of one network run
harness/report.py      builds summary.html from results/
harness/transcripts.py selected agent messages and keyword codings
harness/paper.py       exports numbers, tables, quotes and figures for the write-up
harness/serve.py       serves summary.html on localhost:1306
results/               model outputs (.jsonl) with a .meta.json sidecar each
tests/                 provenance unit test
```
