---
id: B_20261127_1146_browser-screenshots-reach-model
title: Browser screenshots never reached vision models
type: B
author: AlbertYulius
author_email: albert.yulius.ramahalim@huawei.com
modules:
  - openjiuwen/harness
  - openjiuwen/core
files:
  - openjiuwen/harness/tools/browser_move/clients/stdio_client.py
  - openjiuwen/harness/tools/browser_move/clients/streamable_http_client.py
  - openjiuwen/harness/tools/browser_move/playwright_runtime/config.py
  - openjiuwen/harness/tools/browser_move/playwright_runtime/runtime.py
  - openjiuwen/harness/subagents/browser_agent.py
  - openjiuwen/core/single_agent/agents/react_agent.py
  - openjiuwen/core/foundation/tool/base.py
  - tests/unit_tests/harness/tools/browser_move/test_config.py
  - tests/unit_tests/harness/tools/browser_move/test_browser_runtime_tools.py
  - tests/unit_tests/harness/tools/browser_move/test_browser_runtime_rail.py
  - tests/unit_tests/harness/tools/browser_move/test_create_browser_agent.py
  - tests/unit_tests/agent/react_agent/test_multimodal_tool_results.py
  - tests/unit_tests/core/foundation/tool/test_tool_render_for_llm.py
designs:
  - D-20261127-1102-responses-native-support
design_note: The fix is roadmap PR 10 in section 5.6 of the draft design for issue 2898 (not yet merged; branch docs/responses-native-support-design).
---

# Browser screenshots never reached vision models

## Symptom and impact

The browser agent (`create_browser_agent`) offers `browser_take_screenshot` to vision models.
The tool ran, and Playwright MCP saved a correct PNG. But the model never got the image.
The tool message held only `[binary content: image/png]`, and the next request had no image part.
Asked to read a code that appears only inside an image, the model answered "I cannot see the image".

Seen live on 26 Nov 2026 (`develop` at `46014823e`, OpenRouter, `openai/gpt-5.6-luna`, Chat mode):
3 runs out of 3 reproduced it. Related: #2084 (images lost through `tool_call`, fixed by #2085)
and #543 (open feature: let the browser read images in web pages).

Side effect: the browser MCP clients replace the core MCP clients for every stdio and
streamable-HTTP server in the process. So once a browser agent existed, other MCP servers
that set `include_image_content=True` lost their images too.

## Design link and current contract

The draft design `D-20261127-1102-responses-native-support` (section 5.6, "Today: Browser
screenshots", and the PR 10 plan) lists the four causes and this fix. It is not on `develop` yet.
The contracts that apply are `S_05` invariant 11 (tool text comes from `render_for_llm`;
images travel in `multimodal`), `S_04` invariant 13 (screenshot archive rail) and `S_18`
invariant 6 (browser agent wiring). All three are updated in this change.

## Root cause and fix

Four separate causes. Each one alone was enough to lose the image.

1. **The browser MCP clients turned images into text.** `BrowserMoveStdioClient.call_tool` and
   `BrowserMoveStreamableHttpClient.call_tool` had their own result loop that wrote
   `[binary content: <mime>]` and ignored `include_image_content`.
   Fix: both now call the shared `extract_mcp_tool_result_content`, like the core clients.
   Their retry, reconnect and timeout handling is unchanged.
2. **The Playwright MCP config did not ask for images.** `build_playwright_mcp_config` left
   `include_image_content` at `False`. Fix: set it to `True`. Screenshot tools are already
   hidden from models without image support, so images only go to models that can read them.
3. **The rail hid the image from the agent.** `BrowserRuntimeRail._normalize_tool_result`
   unwraps the result to its `data` dict, and `ReActAgent._iter_multimodal_image_items` only read
   `tool_result.data`. Fix: the collector also reads a plain dict that carries `multimodal`.
   `Tool.render_for_llm` renders such a dict without the images, so base64 is not sent as text.
4. **The rail checked image support on the wrong agent.** The tool list used the browser
   DeepAgent (True), but the system prompt and the batch-screenshot check used `ctx.agent`, which
   in model and tool hooks is the inner ReActAgent with no `deep_config` (False). So vision models
   were told not to take screenshots while being offered the tool. Fix: `before_invoke` (which runs
   on the DeepAgent) stores an image-support resolver, and the later hooks use it.

Two more changes keep the fix from wasting context:

- `BrowserRuntimeRail.after_tool_call` takes `multimodal` out of the result while it writes the
  tool message text, receipts and evidence, and puts it back at the end (in a new dict). Without
  this, the 18,176-character base64 image was also sent as tool text.
- `create_browser_agent` adds the existing `MultimodalContextSummarizerRail` (keeps the 3 latest
  screenshots as images), unless the caller already passes one.

Considered and not chosen:

- Keep the `McpToolResult` wrapper inside the rail instead of changing the core collector.
  The rail reads and rewrites the result as a dict in many places, so that change would be
  much larger and riskier. Reading a plain dict in the collector is a one-line change.
- Check image support in `init(agent)`. `init` runs once, so a config hot reload would leave a
  stale answer. `before_invoke` refreshes it on every run.

Behaviour changes worth knowing:

- Text from the browser MCP clients now matches the core clients: blocks are joined with a blank
  line, images become `[image content: <mime>, N base64 chars]` when the flag is off, and the
  streamable-HTTP client now reports MCP `isError` results as failures (the stdio client already did).
- Vision models now get real screenshots, which costs image tokens.

## Regression evidence and scope

- New unit tests, one or more per cause: `test_config.py` (both clients, the config flag),
  `test_browser_runtime_tools.py` (config → client → `MCPTool` → rail → collector),
  `test_browser_runtime_rail.py` (image support from the DeepAgent; no base64 in tool text),
  `test_create_browser_agent.py` (summarizer rail), `test_multimodal_tool_results.py` (plain dict),
  `test_tool_render_for_llm.py` (render without images). 11 of them fail on the old code.
- Wider run on 27 Nov 2026 (Windows, Python 3.11): browser_move, react_agent, core tool, render,
  summarizer, CUA, progressive tool and terminal stream tests: 2221 passed, 67 skipped, 3 failed.
  The 3 failures (`test_dynamic_sort_probe_emits_only_unique_actionable_targets`) are a Windows
  cp1252 encoding error and fail the same way without this change; with `PYTHONUTF8=1` that
  whole file passes.
- Live run on 27 Nov 2026 with this change (`repro_live.py`, OpenRouter, `openai/gpt-5.6-luna`,
  Chat mode): the request after the screenshot had 1 image part, the screenshot tool text
  (456 characters) had no base64, the system prompt said the model can inspect images, and the
  model answered `K7Q-42M`. Verdict: `NO BUG`.
- Not verified: Responses mode (the browser worker cannot be switched to it yet; the image was
  lost before any request was built, so the result should be the same), the streamable-HTTP
  client against a real server, and the separate browser runtime MCP server
  (`build_browser_runtime_mcp_config`), whose configs still leave `include_image_content` off.

## Error-prevention pattern and synced docs

Pattern: a rail that rewrites a tool result must keep `multimodal` out of any text it writes, and
must not hide it from the agent. Recorded in `S_05` invariant 11. Rails that need the DeepAgent's
settings in model or tool hooks must not read them from `ctx.agent` (the inner ReActAgent);
recorded in `S_18` invariant 6. Also updated: `S_04` invariant 13 and the browser agent API docs
(en and zh).

## Follow-ups

- Folding images into the tool result itself, and recording which call made each image, is
  roadmap PR 11 of the same design.
- The browser runtime MCP server configs in `browser_tools.py` do not set `include_image_content`.
  They are a separate path and were left unchanged.
