> **⚠ 全文作废（2026-07-26 18:20）**
>
> 本文件分析的 `naive_tool_caller` 是一个设计错误的 baseline：它给 VLM 装了工具和多轮循环。
> VLM baseline 应当是「给图 + prompt → 直接输出结构化结果」，不该有工具。用户原话：
> "如果让它学会了工具，那它算什么 VLM？"
>
> 因此下面所有关于「VLM 学会 TOOL: 语法」「TOOL+ANSWER 同轮输出」「少样本是否改变对照组」
> 的讨论都建立在错误前提上，**不要用于论文**。`tool_args_overridden` 计数器和
> `_build_prompt` sanitizer 也随之失去意义。
>
> 正确的 VLM baseline 见 `benchmark/vlms.py` + `runners/vlm_runner.py` +
> `runners/arcade_vlm_runner.py`，它一直存在且设计正确。
>
> 保留本文件仅为记录这个错误本身，以免重犯。

# Harness Findings — naive tool-caller, first real VLM run

Run: `runs/harness_smoke2/`, 2026-07-26 03:45.
`naive_tool_caller` × `llava_16_mistral_7b` × `arcade_segmentation`, 2 cases, cuda:5.

This file records **behavioural findings and open decisions**, not results. Both
cases score F1 = 0.0000, and that number is a diagnostic starting point, not a
conclusion. It does not show that naive tool-calling fails; it shows three
independently fixable problems stacked on top of each other, one of which is an
experiment-design question that only the user can settle.

Measured numbers belong in `RESULTS_VERIFIED.md`. Nothing here is paper-ready.

## 1. Zero-shot VLM learned the `TOOL:` protocol

Both cases emitted well-formed tool calls without any example in the prompt:

```
TOOL: segment_vessels(image_path='...', device='cuda:1', method_name='arcade_segmentation_native')
```

`tool_arg_parse_failures = 0` on both. The call syntax (`name(k=v, ...)`) is
learnable from the tool signature block alone. That is a real positive result for
the prompt design — but see §3 for what this counter does *not* measure.

## 2. But the model emits `TOOL:` and `ANSWER:` in the same turn

Both `model_output` values are a tool call immediately followed by a final answer.
The harness loop order is:

1. parse tool calls → execute → append `RESULT: {...}` to history
2. check for `ANSWER:` → if found, `break`

So the tool runs, its result is stored, and the loop exits **before** the result is
fed back to the model. `turns_used = 1` with `max_turns = 8`.

The break logic is correct; the problem is on the model side. The multi-turn budget
is never exercised.

### Case 2 got a valid mask that the model never saw

The successful tool call returned:

```json
{"type": "segmentation_mask", "shape": "(512, 512)", "dtype": "bool",
 "foreground_pixels": 6524, "foreground_fraction": 0.0248870849609375}
```

2.49% foreground. That ratio is the same order of magnitude as the pixel-level
coverage the specialist segmenters report on comparable cases, so the mask is at
least plausible — note that no gold vessel fraction was measured for this case, so
this is a sanity check, not an accuracy claim. It was appended to history and then
discarded by the `break`. The final answer was written without any reference to it.

Both cases have `final_answer_parseable = False`: the model wrote prose, while
`_parse_final_answer` expects structured instances with geometry. So even had the
result been visible, the answer format would still have scored 0.

## 3. `image_path` is silently overridden by the harness

`_execute_tool` replaces whatever path the model emits with the real one:

```python
elif k == "image_path":
    img_path = case.case_dir / "image.png"
    if not img_path.exists():
        images = list(case.case_dir.glob("*.png"))
        if images:
            img_path = images[0]
    converted_args[k] = str(img_path)   # unconditional
```

The VLM emitted a placeholder. Case 2 did not succeed because the model got the
path right; it succeeded because the harness repaired it.

**Consequence**: `tool_arg_parse_failures` measures syntax only (is it `k=v`?), not
semantics (does the path exist? is the method registered?). A model emitting
`/path/to/image` is indistinguishable from one emitting the correct path.

Case 1 failed precisely because the argument with no override logic was wrong:
`method_name='arcade_segmentation_native'` has no `methods/*.toml`, so
`segment_vessels` raised `MethodConfigError`. Syntax valid, semantics wrong.

**Implemented** (`diagnostics["tool_args_overridden"]`): counts how many arguments
the harness rewrote. If Cardiomni teaches the model to address its own data, that
count drops — a measurable capability difference that was previously invisible. Only
counts a genuine mismatch: a model that emits the correct path adds nothing to the
counter. Per-call detail (`model_value` vs `used_value`) is kept in the trace.
Regression tests: `test_placeholder_image_path_counts_as_override` and
`test_correct_image_path_is_not_an_override`.

If arbitrary `method_name` values should be supported, `segment_vessels` must either
tolerate a missing TOML or every callable method must be registered under `methods/`.

## 4. OPEN DECISION — does a few-shot example redefine the baseline?

The obvious fix for §2 is to put a worked two-turn example in the prompt: turn 0
emits only `TOOL:`, turn 1 sees `RESULT:` and then emits `ANSWER:`.

**This is an experiment-design change, not prompt tuning.** Deciding when to call a
tool and when to answer is exactly what the Cardiomni four-stage SOP is meant to
contribute. A naive baseline carrying a full multi-turn example is no longer
unguided; it becomes "few-shot without SOP".

This is the same class of problem as tonight's SOP leak (tool metadata naming
"Stage 4" / "Stage 2" inside the naive prompt), with the sign reversed: that was
accidental leakage, this would be deliberate injection. Both move the control group.

**The user must decide** whether the naive baseline is:

- **(a) zero-shot + tool signatures** — keep it as-is. Report F1 = 0 together with
  `turns_used = 1` and `final_answer_parseable = False`, and state plainly that the
  failure is protocol adherence, not diagnostic capability. Honest, and the widest
  gap to Cardiomni, but a reviewer may object that the baseline is a straw man.
- **(b) few-shot + tool signatures** — add the two-turn example to *every* harness
  including Cardiomni, and say in the paper that the difference is the SOP, not the
  presence of an example. Fairer comparison, smaller measured gap.

Do not treat this as an optimisation to be applied in passing.

## 5. Runtime: 16s on tool success, 282s on tool failure

```
case 1 (MethodConfigError): 282.1s
case 2 (tool succeeded):     16.3s
```

A 17× spread with identical turn counts and comparable output lengths. Earlier
reports of "545.7s, identical for both cases" came from a timer outside the loop
(fixed); an intermediate estimate of 273s/case was inferred from that bug and is
also wrong. The two numbers above are the first trustworthy per-case timings.

Extrapolating 191 cases on one GPU: roughly 51 minutes if every tool call
succeeds, roughly 15 hours if every one fails. The real figure depends on the
effective call rate, so measure it on a subset before scheduling a full sweep.

Both numbers were taken while all 8 H20s sat at 97-100% utilisation, so they are
loaded-machine timings, not clean-room ones.

## Related fix made tonight

`predict` crashed with `AttributeError: 'NoneType' object has no attribute 'get'`
when generation raised, because `final_answer` was left as `None` by design while
the Prediction constructor called `final_answer.get(...)` unconditionally. The
crash also destroyed `diagnostics["generation_error"]`, so the underlying model-side
error was lost. Fixed to `{"raw_output": "", "parsed": None}`: the case scores 0
honestly and the error survives in diagnostics. Without that fix, case 1's
`MethodConfigError` would have been swallowed and both cases would have looked
like unexplained crashes.

Regression test: `test_generation_error_yields_empty_prediction`. Note that
`generate_turn` is imported inside the function, so the patch target is
`vlm_runner.generate_turn`, not `harness_runner.generate_turn` — patching the
latter silently does nothing and the test passes for the wrong reason.
