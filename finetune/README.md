# LocalForge browser model fine-tune

This folder fine-tunes an uncensored 9B model to drive LocalForge's browser well, especially by writing `<js>`
scripts instead of clicking through pages one at a time. It's for Harsh's PC (RTX 3070 8 GB, 16 GB RAM).

**Base model:** `huihui-ai/Huihui-Qwen3.5-9B-abliterated`. It's abliterated, dense, and about 5.6 GB at Q4_K_M,
so it fits entirely in 8 GB of video memory. HauhauCS's 9B only ships as GGUF files, so it can't be trained directly.

**Output:** `Qwen3.5-9B-abliterated-LocalForge-Browser-Q4_K_M.gguf` (plus Q5_K_M and the `mmproj` vision file).
The name keeps "abliterated", so LocalForge lists it.

## Why not the 35B-A3B?
- It's a mixture of experts: only about 3B of its 35B weights work on each token. A dense 9B uses all 9B every token,
  which gives steadier step-by-step reasoning for multi-step browser tasks.
- On an 8 GB card with 16 GB of RAM it only fit as **IQ2_M** (about 2 bits per weight), with most of it running on the CPU.
  That heavy compression costs reasoning quality, and the slower speed leaves less room for context.
- With thinking on, LocalForge keeps half the context free for the reply. Long pages then push older steps out,
  so the model repeats steps. Qwen3.6's long thinking also often hits the app's thinking cap, which costs an extra round.

## How the training data is made (no AI teacher needed)
`sites.py` builds random practice websites in 10 task families:

| Family | What it teaches |
| --- | --- |
| shop_cheapest | open every option, `<note>` each price, then add the cheapest; `<sitesearch>`, `<click text>` |
| table_sum | sum a table with a `<js>` script; if the script finds 0 rows, inspect the HTML and fix the selector |
| paged_scrape | inspect one item's HTML, then one `fetch` + `DOMParser` script that reads every page |
| form_signup | two-step form with `<fill submit="true">` (dropdown, radio, checkboxes) |
| delayed | `<wait for="...">` for content that loads late |
| bulk_inbox | inspect, then a script that ticks every matching checkbox; press Delete; verify with a script |
| api_stock | the number is only in the page's API: `<network find>` then `<network body>` (newest request) |
| infinite_feed | a script that scrolls until nothing new loads, then picks the top item |
| leaderboard | `<extract>`, then rank in `<run lang="python">` (maths in code, not in the head) |
| js_count | count exactly with a script instead of by eye |

A perfect solver for each family plays the task **through LocalForge's real agent loop and real browser**
(`record.py` swaps only the model for the solver). So every example has exactly what the app sends the model:
its full system prompt, tool results, progress notes, history cuts and nudges such as "Check before you choose".
An episode is only kept if the site's own state shows the task was really done (basket, form, inbox) and the
final answer is right. Half the episodes include the short planning thought the app asks for on website tasks.

## Run it
1. Record (any PC with LocalForge's source and Chrome/Edge; about 30 s per episode):
   ```
   python finetune/record.py --app C:\Users\Big_P\Projects\LocalForge-src --out finetune/data/train.jsonl --episodes 450
   python finetune/record.py --app C:\Users\Big_P\Projects\LocalForge-src --out finetune/data/heldout.jsonl --episodes 30 --seed 999
   ```
   The repo already has a recorded set in `data/train.jsonl.gz` and `data/heldout.jsonl.gz`.
2. Train, export and check on Modal (your account; roughly $3-5 of the free credit on an L40S):
   ```
   pip install modal && modal setup        # once
   modal run finetune/modal_train.py
   modal volume get localforge-finetune lf-browser/gguf/ .
   ```
3. Copy the `.gguf` files into `%LOCALAPPDATA%\LocalForge\models`, load the model, and run LocalForge's own browser test:
   `LocalForge.exe --browser-eval "%LOCALAPPDATA%\LocalForge\models\Qwen3.5-9B-abliterated-LocalForge-Browser-Q4_K_M.gguf"`.
   Run it again on the HauhauCS 9B to compare.

`train.py` also runs on its own on any CUDA GPU: `python finetune/train.py --qlora`.
`check_model.py` runs held-out steps and the five refusal prompts against a GGUF with llama-server.
