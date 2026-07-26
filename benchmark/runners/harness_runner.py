"""
Naive tool-calling loop for harness evaluation.

This runner implements the unguided baseline: a reasoning loop with no staged
protocol, no clinical heuristics, and no adaptive replanning. It receives tool
signatures, a task, and a budget, then alternates between model generation and
tool execution until it emits a final answer or runs out of turns.

Why it must stay naive
-----------------------
This is the counterfactual that isolates the contribution of orchestration. If
this loop were given staging hints, fallback strategies, or intermediate checks,
the performance gap between it and Cardiomni would narrow, and the narrowing would
conflate "the base model got better at the task" with "the orchestration got less
relevant". Keeping the prompt fixed ensures that what changes between harness rows
is the sequence of operations, not the quality of generation.

The tool-call format
--------------------
Models emit tool calls as structured text: ``TOOL: <name>(<arg>=<value>, ...)``.
When a call completes, the result is appended to history as
``RESULT: <serialized>``, and generation resumes. A model may call multiple tools
in one turn if it emits multiple ``TOOL:`` lines, though in practice the models
used here (general-purpose VLMs never trained for function-calling) tend to emit
one at a time or confuse the format and produce prose instead.

Errors are kept in the trace rather than silencing them: if a tool raises, the
exception text goes back to the model, and how it reacts is recorded. A robust
harness plans around errors; a weak one retries the same call or gives up.

Trace structure
---------------
Every turn and every tool call is written to ``prediction.diagnostics["trace"]``.
That list is the raw material for analysing orchestration: comparing a naive
trace against Cardiomni's on the same case is the quantitative basis for the
SOP-isolation claim.
"""

from __future__ import annotations

import json
import re
import traceback as tb_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmark.harnesses import HarnessMethod

from benchmark.core import Prediction, Task
from benchmark.io_spec import CaseInput


def _build_prompt(case: CaseInput, tools: dict[str, dict[str, Any]]) -> str:
    """Construct the task + tool description given to the model.

    Tools are described honestly, including their limitations. Hiding that
    segmentation cannot name segments would not make the task harder in a useful
    way; it would just confuse the observable with the true behaviour.
    """
    # Task framing depends on what is being asked
    if case.task is Task.ARCADE_SEGMENTATION:
        task_desc = (
            "Identify every coronary artery segment visible in this XCA frame and "
            "report its SYNTAX id (1..16 + a/b/c subsegments) and bounding box. "
            "Use segment_vessels to localise the vessel tree, then identify which "
            "SYNTAX segment each region corresponds to."
        )
    elif case.task is Task.ARCADE_STENOSIS:
        task_desc = (
            "Find all stenoses in this XCA frame. A stenosis is a segment of artery "
            "that is visibly narrowed compared with the vessel on either side of it. "
            "Report the location of each as a bounding box."
        )
    elif case.task is Task.CARDIOSYNTAX_SCORING:
        task_desc = (
            "Assess the SYNTAX score for this patient, given multiple views of the "
            "coronary tree. The SYNTAX score quantifies disease complexity by summing "
            "weighted contributions from every significant lesion (>=50% stenosis in a "
            "vessel >=1.5mm). Typical range: 0 for normal, 1-22 for low complexity, "
            "23-32 for intermediate, 33+ for high."
        )
    else:
        task_desc = f"Complete the {case.task.value} task."

    tool_lines = []
    for name, meta in tools.items():
        sig = meta.get("signature", f"{name}(...)")
        doc = meta.get("description", "")
        avail = meta.get("available", True)
        boundary = meta.get("capability_boundary", "")
        
        # Sanitize SOP staging vocabulary from tool metadata to avoid contaminating
        # the unguided baseline. Tool metadata should describe the tool's capabilities,
        # not reference the caller's orchestration structure. The proper fix is for
        # the tool layer to remove "Stage N" references entirely; this is a temporary
        # patch to stop the control group from receiving hints about the four-stage
        # SOP that is the independent variable of the harness comparison.
        boundary = re.sub(r'\bStage \d+\b', 'a later stage', boundary, flags=re.IGNORECASE)
        
        tool_lines.append(f"- {sig}")
        if doc:
            tool_lines.append(f"    {doc}")
        if not avail:
            tool_lines.append(f"    WARNING: this tool is unavailable ({meta.get('blocker', 'unknown reason')})")
            alts = meta.get("alternatives")
            if alts:
                tool_lines.append(f"    Alternatives: {'; '.join(alts)}")
        if boundary:
            tool_lines.append(f"    Limitation: {boundary}")
    
    prompt = f"""Task: {task_desc}

Available tools:
{chr(10).join(tool_lines)}

To call a tool, write:
TOOL: <name>(<arg>=<value>, ...)

To provide your final answer, write:
ANSWER: <your response>

Begin."""
    
    return prompt


def _to_chat_messages(
    history: list[dict[str, Any]], attach_image: bool
) -> list[dict[str, Any]]:
    """Convert loop history into HF chat messages.

    The system prompt is folded into the first user message because several of the
    models used here (LLaVA variants) have chat templates that do not accept a
    system role. Dropping it instead would silently remove the tool list, so the
    harness would be measured without ever being told what it could call.
    """
    messages: list[dict[str, Any]] = []
    pending_system: str | None = None

    for entry in history:
        role = entry.get("role")
        text = entry.get("content", "")

        if role == "system":
            pending_system = text
            continue

        if role == "user":
            if pending_system:
                text = f"{pending_system}\n\n{text}" if text else pending_system
                pending_system = None
            content: list[dict[str, Any]] = []
            # The image placeholder belongs only on the first user message; the
            # processor requires exactly as many placeholders as images passed.
            if attach_image and not any(m["role"] == "user" for m in messages):
                content.append({"type": "image"})
            content.append({"type": "text", "text": text})
            messages.append({"role": "user", "content": content})
        else:
            messages.append(
                {"role": role, "content": [{"type": "text", "text": text}]}
            )

    # A system prompt with no following user message still has to be sent.
    if pending_system:
        content = [{"type": "image"}] if attach_image else []
        content.append({"type": "text", "text": pending_system})
        messages.append({"role": "user", "content": content})

    return messages


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract TOOL: calls from model output.

    Returns a list of dicts with 'name' and 'args' keys. Args are kept as strings
    for now; the executor will parse them based on the tool's signature.
    """
    calls = []
    # Match lines like: TOOL: segment_vessels(image_path="/path", device="cuda:1")
    pattern = re.compile(r"TOOL:\s*(\w+)\s*\((.*?)\)", re.IGNORECASE)
    
    for match in pattern.finditer(text):
        name = match.group(1)
        args_str = match.group(2).strip()
        
        # Parse arguments - simple approach for name=value pairs
        args = {}
        arg_parse_failed = False
        if args_str:
            # Handle both quoted and unquoted values
            arg_pattern = re.compile(r'(\w+)=(["\']?)([^,\'"]+)\2')
            for arg_match in arg_pattern.finditer(args_str):
                arg_name = arg_match.group(1)
                arg_value = arg_match.group(3).strip()
                args[arg_name] = arg_value
            # Text between the parentheses that yielded no name=value pair means
            # the model used a format the executor cannot act on (positional args,
            # nested calls, prose). Flagged rather than silently treated as a
            # no-argument call, because "called the tool wrong" and "called the
            # tool with no arguments" are different harness behaviours.
            if not args:
                arg_parse_failed = True
        
        calls.append({"name": name, "args": args, "arg_parse_failed": arg_parse_failed})
    
    return calls


def _execute_tool(
    name: str,
    args: dict[str, str],
    tools: dict[str, Any],
    case: CaseInput,
    device: str,
    use_mock: bool,
) -> dict[str, Any]:
    """Execute one tool call and return the result + metadata.

    Returns a dict with 'result', 'error', 'tool_source' keys. Errors are caught
    rather than propagated so the loop can feed them back to the model.
    """
    result_dict = {"name": name, "args": args, "tool_source": "mock" if use_mock else "real"}
    
    if use_mock:
        # Mock execution for pipeline testing
        if name == "segment_vessels":
            import numpy as np
            result_dict["result"] = {
                "mask_shape": "(512, 512)",
                "foreground_fraction": 0.027,
                "note": "mock output",
            }
        elif name == "quantify_stenosis":
            result_dict["result"] = {
                "reference_diameter_mm": 2.1,
                "mld_mm": 0.24,
                "percent_stenosis": 88.6,
                "severity_class": "critical",
                "note": "mock output",
            }
        elif name == "detect_stenosis":
            result_dict["error"] = "NotImplementedError: DeepCORO-CLIP weights unavailable (mock)"
        else:
            result_dict["error"] = f"unknown tool: {name}"
        return result_dict
    
    # Real execution
    overridden: list[dict[str, str]] = []
    try:
        from algorithms.tools import (
            segment_vessels,
            quantify_stenosis,
            detect_stenosis,
        )
        
        tool_map = {
            "segment_vessels": segment_vessels,
            "quantify_stenosis": quantify_stenosis,
            "detect_stenosis": detect_stenosis,
        }
        
        if name not in tool_map:
            result_dict["error"] = f"unknown tool: {name}"
            return result_dict
        
        func = tool_map[name]
        
        # Convert string args to appropriate types based on tool signature
        converted_args = {}
        for k, v in args.items():
            # Handle device argument
            if k == "device":
                converted_args[k] = device if v in ("auto", "default") else v
            # Handle image_path
            elif k == "image_path":
                # The harness resolves the real image itself rather than trusting the
                # model's path. That keeps a wrong path from failing the case, but
                # it also hides a semantic error: a model that emits a placeholder
                # like '/path/to/image' looks identical to one that emits the
                # correct path. Record the override so "knows where the data is"
                # stays measurable instead of being silently repaired.
                img_path = case.case_dir / "image.png"
                if not img_path.exists():
                    # Fallback to any image in the directory
                    images = list(case.case_dir.glob("*.png"))
                    if images:
                        img_path = images[0]
                if str(img_path) != v:
                    overridden.append(
                        {"arg": k, "model_value": v, "used_value": str(img_path)}
                    )
                converted_args[k] = str(img_path)
            # Handle numeric args
            elif k in ("threshold", "pixel_spacing"):
                try:
                    converted_args[k] = float(v)
                except ValueError:
                    converted_args[k] = v
            # Handle ROI coords as tuple
            elif k == "roi_coords":
                try:
                    # Parse string like "[100,200,300,400]" or "100,200,300,400"
                    coords_str = v.strip("[]()").replace(" ", "")
                    coords = tuple(int(x) for x in coords_str.split(","))
                    converted_args[k] = coords
                except (ValueError, AttributeError):
                    converted_args[k] = v
            else:
                converted_args[k] = v
        
        # Execute the tool
        output = func(**converted_args)
        
        # Serialize the output for feeding back to the model
        if hasattr(output, "shape"):  # numpy array
            import numpy as np
            result_dict["result"] = {
                "type": "segmentation_mask",
                "shape": str(output.shape),
                "dtype": str(output.dtype),
                "foreground_pixels": int(output.sum()),
                "foreground_fraction": float(output.mean()),
            }
        elif isinstance(output, dict):
            result_dict["result"] = output
        elif isinstance(output, list):
            result_dict["result"] = {"instances": output, "count": len(output)}
        else:
            result_dict["result"] = str(output)
            
    except Exception as exc:
        # Capture the full traceback for the diagnostics, and a summary for
        # feeding back to the model
        result_dict["error"] = f"{type(exc).__name__}: {exc}"
        result_dict["traceback"] = tb_module.format_exc()
    
    if overridden:
        result_dict["args_overridden"] = overridden
    return result_dict


def _parse_final_answer(text: str, task: Task) -> dict[str, Any] | None:
    """Extract the final answer from model output, or None if no answer yet.

    The returned dict always carries ``parsed``: the structured value when the
    answer could be turned into something scoreable, or None when only free text
    was recovered. That distinction is the point of this function. An unparseable
    answer and a wrong answer both score zero, and without the flag the results
    cannot tell them apart, which is the same confusion that made a perfect ARCADE
    bbox and a naming-incapable segmenter both read as f1=0.

    No structured extraction is attempted for the ARCADE tasks beyond what the
    harness itself produced. Adding a rescue parser here would do the harness's
    job for it, and the unguided baseline would stop being unguided.
    """
    # Look for ANSWER: marker
    match = re.search(r"ANSWER:\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    
    answer_text = match.group(1).strip()
    
    # Task-specific parsing
    if task in (Task.ARCADE_SEGMENTATION, Task.ARCADE_STENOSIS):
        # Instance lists are left to the scoring stage. `parsed` stays None so the
        # per-case record states plainly that no instances were recovered here.
        return {"raw_output": answer_text, "parsed": None}
    
    elif task is Task.CARDIOSYNTAX_SCORING:
        # Look for a number
        numbers = re.findall(r'\d+(?:\.\d+)?', answer_text)
        if numbers:
            return {
                "score": float(numbers[-1]),
                "raw_output": answer_text,
                "parsed": float(numbers[-1]),
            }
        return {"raw_output": answer_text, "parsed": None}
    
    # Any other task: text captured, nothing structured claimed.
    return {"raw_output": answer_text, "parsed": None}


def predict(
    method: HarnessMethod,
    case: CaseInput,
    output_dir: Path,
    device: str,
) -> Prediction:
    """Run the naive tool-calling loop for one case.

    This is the simplest possible orchestration: prompt → generate → parse tools →
    execute → append results → repeat. No staging, no replanning, no validation
    beyond a turn budget.
    """
    from benchmark.vlms import BY_NAME
    
    # Load the base model
    base_vlm = BY_NAME[method.base_model]
    
    # Gather tool metadata
    tools_meta = {}
    if not method.use_mock_tools:
        from algorithms.tools.vessel_segmentation import segmentation_metadata
        from algorithms.tools.stenosis_detection import is_available, detection_metadata
        
        seg_meta = segmentation_metadata()
        det_available, det_reason = is_available()
        det_meta = detection_metadata()
        
        tools_meta["segment_vessels"] = {
            "signature": "segment_vessels(image_path, device='cuda:1', method_name='coronary_cm_unet_native')",
            "description": "Segment coronary vessels in a 2D XCA frame. Returns a binary mask.",
            "capability_boundary": seg_meta.get("capability_boundary", ""),
            "available": True,
        }
        tools_meta["quantify_stenosis"] = {
            "signature": "quantify_stenosis(image_path, roi_coords, pixel_spacing=0.2, backend='auto')",
            "description": "Measure stenosis severity in a region of interest. Returns diameter and stenosis percentage.",
            "available": True,
        }
        tools_meta["detect_stenosis"] = {
            "signature": "detect_stenosis(image_path, device='cuda:1')",
            "description": "Detect stenoses using DeepCORO-CLIP.",
            "available": det_available,
            "blocker": det_meta.get("blocker", "") if not det_available else "",
            "alternatives": det_meta.get("alternatives", []),
        }
    else:
        # Mock tool metadata
        tools_meta = {
            "segment_vessels": {
                "signature": "segment_vessels(image_path, device='cuda:1')",
                "description": "Segment vessels (mock)",
                "available": True,
            },
            "quantify_stenosis": {
                "signature": "quantify_stenosis(image_path, roi_coords)",
                "description": "Quantify stenosis (mock)",
                "available": True,
            },
            "detect_stenosis": {
                "signature": "detect_stenosis(image_path)",
                "description": "Detect stenoses (mock, always fails)",
                "available": False,
                "blocker": "mock unavailability",
            },
        }
    
    # Build the initial prompt
    system_prompt = _build_prompt(case, tools_meta)
    
    # Initialize the conversation trace
    trace = []
    history = [{"role": "system", "content": system_prompt}]
    
    final_answer = None
    budget_exhausted = False
    generation_error: str | None = None
    tool_arg_parse_failures = 0
    tool_args_overridden = 0
    
    for turn in range(method.max_turns):
        # Generate from the model
        image_path = case.case_dir / "image.png"
        if not image_path.exists():
            images = list(case.case_dir.glob("*.png"))
            image_path = images[0] if images else None
        
        if image_path is None:
            trace.append({
                "turn": turn,
                "error": "no image found in case directory",
            })
            break
        
        turn_entry = {
            "turn": turn,
            "model_input": f"Turn {turn}: {len(history)} history entries",
            "image": str(image_path),
        }
        
        # Generate model output
        if method.use_mock_tools:
            # Mock generation for CPU testing: hardcoded responses
            if turn == 0:
                model_output = "TOOL: segment_vessels(image_path=auto, device=auto)"
            elif turn < method.max_turns - 1:
                model_output = "ANSWER: Based on the segmentation, I identify 3 vessel segments."
            else:
                model_output = "ANSWER: Final answer based on available information."
        else:
            # Real VLM inference through the shared generation entry point. Not a
            # local copy of the processor/generate sequence: see
            # vlm_runner.generate_turn for why duplicating it is a correctness
            # hazard in this repository.
            try:
                from PIL import Image

                from benchmark.runners.vlm_runner import generate_turn

                # The image is attached on the first turn only. Re-sending it every
                # turn would multiply image tokens by the turn count and blow the
                # context window on a long loop; the model keeps the visual content
                # in the conversation history.
                if turn == 0:
                    with Image.open(image_path) as handle:
                        image = handle.convert("RGB")
                        messages = _to_chat_messages(history, attach_image=True)
                        model_output = generate_turn(
                            base_vlm, device, messages, [image]
                        )
                else:
                    messages = _to_chat_messages(history, attach_image=False)
                    model_output = generate_turn(base_vlm, device, messages, None)

            except Exception as exc:
                # A generation failure ends the case with an explicit marker rather
                # than a fabricated answer. Recorded at the top level of
                # diagnostics so a sweep-wide count is visible without reading
                # every trace.
                turn_entry["generation_error"] = f"{type(exc).__name__}: {exc}"
                turn_entry["model_output"] = None
                trace.append(turn_entry)
                generation_error = f"{type(exc).__name__}: {exc}"
                break
        
        turn_entry["model_output"] = model_output
        
        # Parse tool calls
        tool_calls = _parse_tool_calls(model_output)
        if tool_calls:
            turn_entry["tool_calls"] = []
            for call in tool_calls:
                if call.get("arg_parse_failed"):
                    tool_arg_parse_failures += 1
                    
                result = _execute_tool(
                    call["name"],
                    call["args"],
                    tools_meta,
                    case,
                    device,
                    method.use_mock_tools,
                )
                turn_entry["tool_calls"].append(result)
                tool_args_overridden += len(result.get("args_overridden", []))
                
                # Append result to history for next turn
                if "error" in result:
                    history.append({
                        "role": "system",
                        "content": f"RESULT: Tool {result['name']} failed: {result['error']}"
                    })
                else:
                    history.append({
                        "role": "system",
                        "content": f"RESULT: {json.dumps(result['result'])}"
                    })
        
        # Check for final answer
        answer_dict = _parse_final_answer(model_output, case.task)
        if answer_dict is not None:
            final_answer = answer_dict
            turn_entry["final_answer"] = True
            # Item 3: flag whether the answer text can be parsed into the expected
            # structure. Both "gave an answer but it's wrong" and "gave text we
            # cannot parse at all" score zero, but they reflect different harness
            # behaviours and must be distinguishable in the results.
            turn_entry["final_answer_parseable"] = answer_dict.get("parsed") is not None
            trace.append(turn_entry)
            break
        
        trace.append(turn_entry)
        history.append({"role": "assistant", "content": model_output})
    
    # A generation error leaves final_answer as None, so the case produces an
    # empty prediction rather than a fabricated one. Recorded at the top level so
    # sweep-wide counts do not require reading every trace.
    if generation_error:
        budget_exhausted = False  # Distinguishes "ran out of budget" from "crashed"
        # The loop broke before producing an answer. An empty answer keeps the case
        # scoreable as a zero without inventing content: the failure is recorded in
        # diagnostics["generation_error"], not disguised as a model response.
        final_answer = {"raw_output": "", "parsed": None}
    elif final_answer is None:
        budget_exhausted = True
        final_answer = {"raw_output": "Budget exhausted, no final answer produced"}
    
    # Convert the final answer to a Prediction
    # This is task-dependent
    pred_kwargs = {
        "case_id": case.case_id,
        "task": case.task,
        "diagnostics": {
            "harness": method.name,
            "base_model": method.base_model,
            "turns_used": len(trace),
            "max_turns": method.max_turns,
            "budget_exhausted": budget_exhausted,
            "generation_error": generation_error,
            "tool_arg_parse_failures": tool_arg_parse_failures,
            # Count of arguments the harness rewrote because the model's value was
            # unusable. Distinct from tool_arg_parse_failures, which only sees
            # syntax: a syntactically perfect call carrying a placeholder path
            # raises this counter but not that one.
            "tool_args_overridden": tool_args_overridden,
            "tool_source": "mock" if method.use_mock_tools else "real",
            "trace": trace,
        },
        "raw_output": final_answer.get("raw_output", ""),
    }
    
    if case.task is Task.CARDIOSYNTAX_SCORING:
        pred_kwargs["score"] = final_answer.get("score", 0.0)
    elif case.task in (Task.ARCADE_SEGMENTATION, Task.ARCADE_STENOSIS):
        # Would need proper parsing here
        pred_kwargs["instances"] = []
    
    return Prediction(**pred_kwargs)
