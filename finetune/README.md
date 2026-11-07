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

## Run it (everything on the PC)
Training runs in WSL2 Ubuntu on the RTX 3070. Close LocalForge first so all 8 GB of video memory are free.
1. One-time setup (Python GPU tools, plus llama.cpp's converter at LocalForge's build b10549):
   `wsl -u root bash finetune/local/setup_wsl.sh`
2. Quick test, about 15 min (GPU memory and speed): `wsl bash finetune/local/run_wsl.sh test`
3. Training (several hours; if it stops, run it again and it carries on): `wsl bash finetune/local/run_wsl.sh train`
4. GGUF export (Q4_K_M, Q5_K_M and the mmproj vision file), copied straight into LocalForge's models folder:
   `wsl bash finetune/local/run_wsl.sh export`
5. Compare it with LocalForge's own browser test, then run the same test on the HauhauCS 9B:
   `LocalForge.exe --browser-eval "%LOCALAPPDATA%\LocalForge\models\Qwen3.5-9B-abliterated-LocalForge-Browser-Q4_K_M.gguf"`
   `check_model.py` also checks held-out steps and the five refusal prompts with LocalForge's llama-server.

**How it fits in 8 GB:** the model is loaded in 4-bit (QLoRA). The 2 GB embedding table and the vision tower stay in
system RAM, and the output layer is 4-bit too. Logits are only computed where there's a loss.
Each website task is trained as one sequence: the app's last call of the episode, with the loss on every reply
in it, plus the planning steps where the app asked for thinking. That's 665 sequences and 3.4M tokens per epoch,
about 5x less compute than one sequence per call. The adapter is merged one safetensors shard at a time, so
16 GB of RAM is enough.

To record more data: `python finetune/record.py --app C:\Users\Big_P\Projects\LocalForge-src --out finetune/data/more.jsonl --episodes 200 --seed 7`
(Chrome or Edge is needed, and each episode takes about 30 s).
