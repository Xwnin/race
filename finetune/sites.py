"""Practice websites and perfect solvers for fine-tuning a LocalForge browser model.

Each task family makes a random small website, a task in plain English, a solver that writes exactly the
replies a good LocalForge model should write (the app's own tags, short plans, <note>s, <js> scripts), and a
check that the task really got done. record.py runs the solver through LocalForge's real agent loop and
browser, so every training example shows the model what the app really sends it.

A solver is a generator: it receives the latest message the app sent the model and yields the next reply,
either a string or Reply(text, thought) when the app asked the model to think first.
"""
import json
import random
import re
import urllib.parse
from dataclasses import dataclass, field


@dataclass
class Reply:
    text: str
    thought: str = ""


@dataclass
class Task:
    family: str
    prompt: str
    routes: dict                      # path -> fn(query: dict, form: dict, state: dict) -> (body, status, content_type)
    solve: object                     # fn(base_url) -> generator
    check: object                     # fn(state, final_answer) -> bool
    state: dict = field(default_factory=dict)


FIRST = ["Alex", "Sam", "Priya", "Tom", "Mei", "Omar", "Lena", "Diego", "Aisha", "Jonas", "Kofi", "Sara", "Ivan", "Noor", "Luca"]
LAST = ["Doe", "Patel", "Kim", "Okafor", "Novak", "Silva", "Berg", "Haddad", "Ito", "Reyes", "Clarke", "Moreau"]
COUNTRIES = ["France", "Canada", "Japan", "Brazil", "Kenya", "Norway", "India", "Mexico", "Spain", "Australia"]
SHOPS = ["KitchenHub", "HomeNest", "GadgetBarn", "TechTrove", "CasaPlus", "DailyDeals"]
GOODS = {"kettle": ["1.7L", "1.5L", "compact 1.0L", "deluxe 1.7L", "travel 0.5L", "glass 1.8L"],
         "toaster": ["2-slice", "4-slice", "retro 2-slice", "long-slot", "compact"],
         "blender": ["600W", "1000W", "personal", "pro 1200W", "glass jug"],
         "headphones": ["wireless", "over-ear", "noise cancelling", "sport", "studio"],
         "desk lamp": ["LED", "clip-on", "dimmable", "USB", "architect"],
         "backpack": ["20L", "30L", "laptop", "hiking 40L", "roll-top"]}
COLOURS = ["blue", "red", "black", "white", "green", "silver"]


def nav(n=30):
    return "".join(f'<a href="/help/{i}">Help topic {i}</a> ' for i in range(n))


def page(title, body, head="", footer=True):
    return (f"<!doctype html><html><head><meta charset=utf-8><title>{title}</title>{head}</head><body>"
            f"<header><a href=\"/\">Home</a> <a href=\"/about\">About</a> <a href=\"/contact\">Contact</a></header>"
            f"<h1>{title}</h1>{body}" + (f"<footer>{nav()}</footer>" if footer else "") + "</body></html>")


def html(body, code=200):
    return body, code, "text/html; charset=utf-8"


def jsonr(obj, code=200):
    return json.dumps(obj), code, "application/json"


def money(x):
    return f"{x:.2f}"


def help_route(q, f, s):
    return html(page("Help", "<p>Nothing useful here.</p>"))


def base_routes():
    return {"/about": help_route, "/contact": help_route, "/help/*": help_route}


def plan(*steps):
    return "Plan:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))


def find_num(obs, pattern):
    """Element number of the first element line matching pattern, e.g. find_num(obs, r'button "Delete"')."""
    m = re.search(r"\[(\d+)\][^\n]*" + pattern, obs, re.I)
    return m.group(1) if m else None


def script_json(obs):
    """The JSON a <js> script or <network body> returned, parsed (the app prints it with \\u escapes)."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"[{\[]", obs):
        try:
            v = dec.raw_decode(obs[m.start():])[0]
        except ValueError:
            continue
        if v:
            return v
    return None


def inspect_js(text):
    """A small first script: the HTML of the element that holds a piece of visible text, to learn the real class names."""
    return ("<js>\nconst hits = [...document.querySelectorAll('body *')].filter(e => e.children.length && e.innerText.includes("
            + json.dumps(text) + "));\nconst el = hits[hits.length - 1];\n"
            "({tag: el.tagName, cls: el.className, parent: el.parentElement.tagName + '.' + el.parentElement.className,"
            " count: document.querySelectorAll(el.tagName + (el.className ? '.' + el.className.split(' ')[0] : '')).length,"
            " html: el.outerHTML.slice(0, 500)})\n</js>")


# ---------------------------------------------------------------- A: cheapest item, prices only on item pages
def shop_cheapest(rng: random.Random):
    shop, kind = rng.choice(SHOPS), rng.choice(list(GOODS))
    colour = rng.choice(COLOURS)
    variants = rng.sample(GOODS[kind], rng.randint(3, min(5, len(GOODS[kind]))))
    items = []
    for i, v in enumerate(variants):
        items.append({"id": f"p{i + 1}", "name": f"{colour.title()} {kind} {v}", "price": round(rng.uniform(12, 90), 2)})
    others = [{"id": f"x{i}", "name": f"{c.title()} {kind} {rng.choice(GOODS[kind])}", "price": round(rng.uniform(5, 15), 2)}
              for i, c in enumerate(rng.sample([c for c in COLOURS if c != colour], 2))]
    allp = items + others
    prices = sorted(p["price"] for p in items)
    if prices[0] == prices[1]:
        items[0]["price"] += 1.37
    best = min(items, key=lambda p: p["price"])
    st = {"basket": []}

    def home(q, f, s):
        return html(page(f"{shop} shop", '<form action="/search"><input name="q" placeholder="Search products" aria-label="Search">'
                                          '<button>Search</button></form><p>Welcome! Use the search box to find products.</p>'))

    def search(q, f, s):
        words = q.get("q", [""])[0].lower().split()
        hits = [p for p in allp if all(w in p["name"].lower() for w in words)]
        rng2 = random.Random(len(words))
        rng2.shuffle(hits)
        rows = "".join(f'<li><a href="/product/{p["id"]}">{p["name"]}</a></li>' for p in hits) or "<li>No products found</li>"
        return html(page("Search results", f"<p>Prices are shown on each product page.</p><ul>{rows}</ul>"))

    def product(q, f, s, pid=None):
        p = next((p for p in allp if p["id"] == pid), None)
        if not p:
            return html(page("Not found", ""), 404)
        return html(page(p["name"], f'<p class=price>Price: £{money(p["price"])}</p><a href="#" onclick="return false">Add</a>'
                         + "<p>Great product. Customers love it.</p>" * 25 +
                         f'<form method=post action="/basket/add"><input type=hidden name=id value={p["id"]}>'
                         f'<button>Add to basket</button></form><a href="/basket">View basket</a>'))

    def add(q, f, s):
        s["basket"].append(f.get("id", [""])[0])
        return html(page("Added to basket", '<p>Item added.</p><a href="/basket">View basket</a>'))

    def basket(q, f, s):
        its = [p for p in allp if p["id"] in s["basket"]]
        rows = "".join(f"<li>{p['name']} - £{money(p['price'])}</li>" for p in its) or "<li>Your basket is empty</li>"
        return html(page("Your basket", f"<ul>{rows}</ul><p>Basket total: £{money(sum(p['price'] for p in its))}</p>"))

    routes = {**base_routes(), "/": home, "/search": search, "/basket": basket, "/basket/add": add}
    for p in allp:
        routes[f"/product/{p['id']}"] = (lambda pid: lambda q, f, s: product(q, f, s, pid))(p["id"])
    query = f"{colour} {kind}"

    def solve(base):
        obs = yield Reply(plan(f"Search the shop for {query}", "Open every result and note its price",
                               "Open the cheapest one and add it to the basket", "Open the basket and read the total")
                          + f'\n<browse url="{base}/"/>',
                          thought=f"Prices are probably only on product pages, so I must open every {query} before choosing.")
        obs = yield f'<sitesearch query="{query}"/>'
        names = re.findall(r'link "([^"]+)" -> \S*/product/', obs)
        names = [n for n in names if n.lower().startswith(colour)]
        seen = {}
        for n in names:
            obs = yield f'<click text="{n}"/>'
            m = re.search(r"Price: £([\d.]+)", obs)
            seen[n] = float(m.group(1)) if m else 1e9
            if n != names[-1]:
                obs = yield f'<note>{n}: £{m.group(1) if m else "?"}</note>\n<back/>'
            else:
                note = f'<note>{n}: £{m.group(1) if m else "?"}</note>'
        cheapest = min(seen, key=seen.get)
        if cheapest == names[-1]:
            obs = yield f"{note}\nAll {len(names)} prices noted. The cheapest is {cheapest}, and its page is open.\n<click text=\"Add to basket\"/>"
        else:
            obs = yield f"{note}\nAll {len(names)} prices noted. The cheapest is {cheapest} (£{money(seen[cheapest])}).\n<back/>"
            obs = yield f'<click text="{cheapest}"/>'
            obs = yield '<click text="Add to basket"/>'
        obs = yield '<click text="View basket"/>'
        m = re.search(r"Basket total: £([\d.]+)", obs)
        yield f"I added the cheapest {query}, the {cheapest} (£{money(seen[cheapest])}), to your basket. The basket total is £{m.group(1) if m else '?'}."

    return Task("shop_cheapest", f"Go to {{base}}/ and add the cheapest {query} in the shop to my basket, then go to the basket and tell me the basket total.",
                routes, solve, lambda s, ans: s["basket"] == [best["id"]] and money(best["price"]) in ans, st)


# ---------------------------------------------------------------- B: table maths with a <js> script
def table_sum(rng: random.Random):
    custs = [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(6)]
    statuses = ["Shipped", "Pending", "Cancelled", "Refunded"]
    rows = [{"id": f"#{1000 + i}", "cust": rng.choice(custs), "status": rng.choice(statuses), "amount": round(rng.uniform(5, 400), 2)}
            for i in range(rng.randint(25, 90))]
    who, status = rng.choice(custs), rng.choice(statuses[:2])
    want = [r for r in rows if r["cust"] == who and r["status"] == status]
    if not want:
        rows[0].update(cust=who, status=status)
        want = [rows[0]]
    total = round(sum(r["amount"] for r in want), 2)
    grid = rng.random() < 0.5          # div grid instead of a table: the first script finds nothing and must be fixed
    if grid:
        body = '<div class="orders">' + "".join(
            f'<div class="order-row"><span class="c-id">{r["id"]}</span><span class="c-cust">{r["cust"]}</span>'
            f'<span class="c-status">{r["status"]}</span><span class="c-amt">${money(r["amount"])}</span></div>' for r in rows) + "</div>"
    else:
        body = ("<table><thead><tr><th>Order</th><th>Customer</th><th>Status</th><th>Amount</th></tr></thead><tbody>" +
                "".join(f'<tr><td>{r["id"]}</td><td>{r["cust"]}</td><td>{r["status"]}</td><td>${money(r["amount"])}</td></tr>' for r in rows)
                + "</tbody></table>")
    routes = {**base_routes(), "/orders": lambda q, f, s: html(page("Orders", "<p>All orders this quarter.</p>" + body))}
    js_table = (f"const rows = [...document.querySelectorAll('table tbody tr')].map(tr => [...tr.cells].map(td => td.innerText.trim()));\n"
                f"const hits = rows.filter(r => r[1] === {json.dumps(who)} && r[2] === {json.dumps(status)});\n"
                "const total = hits.reduce((t, r) => t + parseFloat(r[3].replace(/[^0-9.]/g, '')), 0);\n"
                "({rows: rows.length, matches: hits.length, ids: hits.map(r => r[0]), total: total.toFixed(2)})")
    js_grid = (f"const rows = [...document.querySelectorAll('.order-row')].map(d => [...d.children].map(c => c.innerText.trim()));\n"
               f"const hits = rows.filter(r => r[1] === {json.dumps(who)} && r[2] === {json.dumps(status)});\n"
               "const total = hits.reduce((t, r) => t + parseFloat(r[3].replace(/[^0-9.]/g, '')), 0);\n"
               "({rows: rows.length, matches: hits.length, ids: hits.map(r => r[0]), total: total.toFixed(2)})")

    def solve(base):
        obs = yield Reply(plan("Open the orders page", f"Use a script to sum every {status} order for {who}", "Report the total")
                          + f'\n<browse url="{base}/orders"/>',
                          thought="Many rows: summing by hand is error-prone, so a <js> script should read and sum them.")
        obs = yield f"<js>\n{js_table}\n</js>"
        if '"rows": 0' in obs:
            obs = yield ("0 rows: there's no table, so my selector is wrong. Let me check how the rows are built.\n"
                         "<js>\nconst el = document.querySelector('h1').nextElementSibling.nextElementSibling;\n"
                         "({tag: el.tagName, cls: el.className, first: el.firstElementChild.outerHTML.slice(0, 300)})\n</js>")
            obs = yield f"The rows are div.order-row with one span per column. Fixed script:\n<js>\n{js_grid}\n</js>"
        d = script_json(obs) or {}
        yield f"{who} has {d.get('matches', '?')} {status} orders ({', '.join(d.get('ids', []))}), totalling ${d.get('total', '?')}."

    return Task("table_sum", f"Open {{base}}/orders and tell me the total amount of all {status} orders for {who}.",
                routes, solve, lambda s, ans: money(total) in ans)


# ---------------------------------------------------------------- C: many pages, one fetch+DOMParser script
def paged_scrape(rng: random.Random):
    cities = ["Remote", "London", "Berlin", "Austin", "Toronto", "Lisbon"]
    roles = ["Backend Engineer", "Data Analyst", "QA Tester", "DevOps Engineer", "Frontend Developer", "ML Engineer", "Support Lead"]
    jobs = [{"title": f"{rng.choice(roles)}", "co": f"{rng.choice(LAST)} {rng.choice(['Labs', 'Systems', 'Tech', 'Group'])}",
             "city": rng.choice(cities), "pay": rng.randrange(30, 140) * 1000} for _ in range(rng.randint(30, 70))]
    per = 10
    npages = (len(jobs) + per - 1) // per
    city, floor = rng.choice(cities[:3]), rng.choice([60000, 80000, 100000])
    want = [j for j in jobs if j["city"] == city and j["pay"] >= floor]

    def listing(q, f, s):
        p = int(q.get("page", ["1"])[0])
        chunk = jobs[(p - 1) * per: p * per]
        cards = "".join(f'<article class="job"><h3>{j["title"]}</h3><p class="co">{j["co"]}</p><p class="loc">{j["city"]}</p>'
                        f'<p class="pay">£{j["pay"]:,} a year</p></article>' for j in chunk)
        pager = " ".join(f'<a href="/jobs?page={i}">{i}</a>' for i in range(1, npages + 1))
        nxt = f'<a href="/jobs?page={p + 1}">Next</a>' if p < npages else ""
        return html(page(f"Jobs - page {p} of {npages}", cards + f"<nav class=pager>{pager} {nxt}</nav>"))

    routes = {**base_routes(), "/jobs": listing}
    script = ("const out = [];\n"
              f"for (let p = 1; p <= {{N}}; p++) {{\n"
              "  const html = await (await fetch(`/jobs?page=${p}`)).text();\n"
              "  const doc = new DOMParser().parseFromString(html, 'text/html');\n"
              "  doc.querySelectorAll('article.job').forEach(a => out.push({\n"
              "    title: a.querySelector('h3').innerText, company: a.querySelector('.co').innerText,\n"
              "    city: a.querySelector('.loc').innerText, pay: +a.querySelector('.pay').innerText.replace(/[^0-9]/g, '')}));\n"
              "}\n"
              f"const hits = out.filter(j => j.city === {json.dumps(city)} && j.pay >= {floor});\n"
              "({scanned: out.length, matches: hits.length, jobs: hits.map(j => `${j.title} at ${j.company} (£${j.pay.toLocaleString('en-GB')})`)})")

    def solve(base):
        obs = yield Reply(plan("Open the first jobs page to see how many pages there are", "Check one job's HTML for its class names",
                               f"Run one script that fetches every page and keeps the {city} jobs paying £{floor:,}+", "List them")
                          + f'\n<browse url="{base}/jobs?page=1"/>',
                          thought="Clicking through every page is slow; one script can fetch them all.")
        m = re.search(r"page 1 of (\d+)", obs)
        n = m.group(1) if m else str(npages)
        obs = yield f"There are {n} pages. Before scripting, I'll check how one job is built in the HTML.\n" + inspect_js(jobs[0]["co"])
        obs = yield (f"Each job is an article.job with h3 (title), .co (company), .loc (city) and .pay. One script for all {n} pages:\n"
                     f"<js>\n{script.replace('{N}', n)}\n</js>")
        d = script_json(obs) or {}
        found, sc = d.get("jobs", []), d.get("scanned", "?")
        if found:
            yield (f"I checked all {sc} jobs on {n} pages. {len(found)} {city} jobs pay £{floor:,} or more:\n"
                   + "\n".join(f"- {j}" for j in found))
        else:
            yield f"I checked all {sc} jobs on {n} pages. None in {city} pays £{floor:,} or more."

    def check(s, ans):
        return all(f"£{j['pay']:,}" in ans and j["co"] in ans for j in want) and (want or "None" in ans)

    return Task("paged_scrape", f"Go to {{base}}/jobs and find every job in {city} that pays at least £{floor:,} a year. Check all the pages, then list them.",
                routes, solve, check)


# ---------------------------------------------------------------- D: two-step form with <fill>
def form_signup(rng: random.Random):
    name, email = f"{rng.choice(FIRST)} {rng.choice(LAST)}", f"{rng.choice(FIRST).lower()}{rng.randint(1, 99)}@example.com"
    country, plan_ = rng.choice(COUNTRIES), rng.choice(["Basic", "Plus", "Pro"])
    news = rng.random() < 0.5
    code = f"{rng.choice(['LF', 'AC', 'ZX'])}-{rng.randint(10000, 99999)}"
    st = {"signup": None}

    def s1(q, f, s):
        return html(page("Create account - step 1 of 2", '<form action="/signup2"><label>Full name <input name=name></label>'
                                                         '<label>Email <input name=email type=email></label><button>Next</button></form>'))

    def s2(q, f, s):
        hidden = "".join(f'<input type=hidden name={k} value="{v[0]}">' for k, v in q.items())
        opts = "".join(f"<option>{c}</option>" for c in COUNTRIES)
        plans = "".join(f'<label><input type=radio name=plan value={p}> {p}</label>' for p in ["Basic", "Plus", "Pro"])
        return html(page("Create account - step 2 of 2", f'<form method=post action="/signup/done">{hidden}'
                         f'<label>Country <select name=country><option>Choose…</option>{opts}</select></label>{plans}'
                         '<label><input type=checkbox name=news> Send me the newsletter</label>'
                         '<label><input type=checkbox name=terms> I accept the terms</label><button>Create account</button></form>'))

    def done(q, f, s):
        s["signup"] = {k: v[0] for k, v in f.items()}
        if "terms" not in f:
            return html(page("Error", "<p>You must accept the terms. Go back and tick the box.</p>"))
        return html(page("Account created", f"<p>Welcome! Your confirmation code is {code}.</p>"))

    routes = {**base_routes(), "/signup": s1, "/signup2": s2, "/signup/done": done}

    def solve(base):
        obs = yield Reply(plan("Open the sign-up page", "Fill step 1 (name, email) and go on",
                               "Fill step 2 (country, plan, newsletter, terms) and create the account", "Read the confirmation code")
                          + f'\n<browse url="{base}/signup"/>',
                          thought="Two-step form: <fill> each step by field names, submit, check the next page.")
        obs = yield f'<fill submit="true">\nFull name: {name}\nEmail: {email}\n</fill>'
        obs = yield ("Step 2 is open.\n<fill submit=\"true\">\n"
                     f"Country: {country}\n{plan_}: yes\nSend me the newsletter: {'yes' if news else 'no'}\nI accept the terms: yes\n</fill>")
        m = re.search(r"confirmation code is ([A-Z]{2}-\d+)", obs)
        yield f"The account for {name} is created. Your confirmation code is {m.group(1) if m else '?'}."

    def check(s, ans):
        d = s["signup"] or {}
        return (d.get("name") == name and d.get("email") == email and d.get("country") == country and d.get("plan") == plan_
                and ("news" in d) == news and "terms" in d and code in ans)

    extra = f", plan {plan_}" + (", and sign me up to the newsletter" if news else ", no newsletter")
    return Task("form_signup", f"Go to {{base}}/signup and create an account: name {name}, email {email}, country {country}{extra}. "
                               "Accept the terms, then tell me the confirmation code.", routes, solve, check, st)


# ---------------------------------------------------------------- E: content that appears later
def delayed(rng: random.Random):
    inc, limit, region = f"INC-{rng.randint(1000, 9999)}", rng.choice([300, 600, 900, 1200]), rng.choice(["EU-West", "US-East", "Asia-1"])
    ms = rng.choice([2500, 3500, 5000])

    def status(q, f, s):
        return html(page("Service status", '<div id=s>Loading status…</div><script>setTimeout(() => document.getElementById("s").innerHTML = '
                         f'"<p>Region {region}: degraded. Incident ID: {inc}. API rate limit: {limit} requests per minute.</p>", {ms})</script>'))

    def solve(base):
        obs = yield Reply(f'<browse url="{base}/status"/>', thought="Open the status page and read the incident.")
        if "Incident ID" not in obs:
            obs = yield 'The status is still loading. I\'ll wait for it.\n<wait for="Incident ID"/>'
        m, l = re.search(r"Incident ID: (INC-\d+)", obs), re.search(r"rate limit: (\d+)", obs)
        yield f"The incident ID is {m.group(1) if m else '?'} and the API rate limit is {l.group(1) if l else '?'} requests per minute."

    return Task("delayed", f"Open {{base}}/status and tell me the incident ID and the API rate limit.",
                {**base_routes(), "/status": status}, solve, lambda s, ans: inc in ans and str(limit) in ans)


# ---------------------------------------------------------------- F: bulk action with a <js> script
def bulk_inbox(rng: random.Random):
    senders = [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(5)] + ["Promo Bot", "Weekly Digest"]
    subj = ["Invoice", "Meeting notes", "Quick question", "Your order", "Big sale!", "Newsletter", "Reminder", "Photos"]
    msgs = [{"id": i, "from": rng.choice(senders), "subject": rng.choice(subj)} for i in range(rng.randint(15, 40))]
    target = rng.choice(["Promo Bot", "Weekly Digest", senders[0]])
    if not any(m["from"] == target for m in msgs):
        msgs[0]["from"] = target
    st = {"msgs": [dict(m) for m in msgs]}

    def inbox(q, f, s):
        rows = "".join(f'<li class="mail"><label><input type=checkbox name=sel value={m["id"]}> <b class="from">{m["from"]}</b> '
                       f'<span class="subj">{m["subject"]}</span></label></li>' for m in s["msgs"])
        return html(page(f"Inbox ({len(s['msgs'])})", f'<form method=post action="/delete"><button>Delete selected</button><ul>{rows}</ul></form>'))

    def delete(q, f, s):
        gone = set(f.get("sel", []))
        s["msgs"] = [m for m in s["msgs"] if str(m["id"]) not in gone]
        return html(page("Deleted", f'<p>{len(gone)} messages deleted.</p><a href="/inbox">Back to inbox</a>'))

    want_left = len([m for m in msgs if m["from"] != target])
    script = (f"const boxes = [...document.querySelectorAll('li.mail')].filter(li => li.querySelector('.from').innerText.trim() === {json.dumps(target)});\n"
              "boxes.forEach(li => li.querySelector('input[type=checkbox]').checked = true);\n"
              "boxes.length")

    def solve(base):
        obs = yield Reply(plan("Open the inbox", f"Tick every message from {target} with a script", "Press Delete selected", "Check the inbox again")
                          + f'\n<browse url="{base}/inbox"/>',
                          thought="Ticking dozens of boxes one by one is slow; a script can tick them all, then I press the button.")
        obs = yield "I'll check how one message row is built first.\n" + inspect_js(msgs[0]["subject"])
        obs = yield (f"Each message is an li.mail with a checkbox and the sender in b.from. Script to tick every one from {target}:\n"
                     f"<js>\n{script}\n</js>")
        n = re.search(r"Script result:\n(\d+)", obs)
        obs = yield f"{n.group(1) if n else 'The'} messages from {target} are ticked.\n<click text=\"Delete selected\"/>"
        obs = yield '<click text="Back to inbox"/>'
        left = re.search(r"Inbox \((\d+)\)", obs)
        chk = f"const left = [...document.querySelectorAll('li.mail .from')].filter(b => b.innerText.trim() === {json.dumps(target)}).length;\nleft"
        obs = yield f"<js>\n{chk}\n</js>"
        yield f"Done: I deleted every message from {target}. None are left, and {left.group(1) if left else '?'} messages remain in the inbox."

    return Task("bulk_inbox", f"Go to {{base}}/inbox and delete every message from {target}, then tell me how many messages are left.",
                {**base_routes(), "/inbox": inbox, "/delete": delete}, solve,
                lambda s, ans: not any(m["from"] == target for m in s["msgs"]) and len(s["msgs"]) == want_left and str(want_left) in ans, st)


# ---------------------------------------------------------------- G: data only in the page's API response
def api_stock(rng: random.Random):
    kind = rng.choice(list(GOODS))
    items = [{"sku": f"SKU-{rng.randint(100, 999)}{chr(65 + i)}", "name": f"{rng.choice(COLOURS).title()} {kind} {v}",
              "stock": rng.randint(0, 250), "warehouse": rng.choice(["Leeds", "Bristol", "Glasgow"])}
             for i, v in enumerate(rng.sample(GOODS[kind], 4))]
    pick = rng.choice(items)
    js = ("<div id=list>Loading…</div><script>fetch('/api/inventory?v=2').then(r => r.json()).then(d => "
          "document.getElementById('list').innerHTML = d.items.map(i => `<p>${i.name}: ${i.stock > 0 ? 'In stock' : 'Out of stock'}</p>`).join(''))</script>")

    def inv(q, f, s):
        return html(page("Inventory", "<p>Live stock levels.</p>" + js))

    def api(q, f, s):
        return jsonr({"items": items, "updated": "2026-11-07T09:00:00Z"})

    def solve(base):
        obs = yield Reply(f'<browse url="{base}/inventory"/>', thought="Open the page first and see what it shows.")
        obs = yield ("The page only says In stock / Out of stock, not the number. It loads its data from an API, so I'll read that.\n"
                     '<network find="inventory"/>')
        n = re.findall(r"^(\d+)\. GET [^\n]*/api/inventory", obs, re.M)     # newest is last
        obs = yield f'<network body="{n[-1] if n else 1}"/>'
        d = script_json(obs) or {}
        it = next((i for i in d.get("items", []) if i.get("name") == pick["name"]), {})
        yield f"{pick['name']} ({it.get('sku', '?')}) has {it.get('stock', '?')} in stock, in the {it.get('warehouse', '?')} warehouse."

    return Task("api_stock", f"Open {{base}}/inventory and find the exact number in stock for {pick['name']}, and which warehouse it's in.",
                {**base_routes(), "/inventory": inv, "/api/inventory": api}, solve,
                lambda s, ans: str(pick["stock"]) in ans and pick["warehouse"] in ans)


# ---------------------------------------------------------------- H: infinite scroll, loop in <js>
def infinite_feed(rng: random.Random):
    tags = ["python", "rust", "javascript", "go", "sql", "security"]
    posts = [{"t": f"Post {i + 1}: notes on {rng.choice(tags)}", "tag": rng.choice(tags), "likes": rng.randint(0, 500)} for i in range(rng.randint(40, 90))]
    tag = rng.choice(tags)
    want = [p for p in posts if p["tag"] == tag]
    if not want:
        posts[0]["tag"] = tag
        want = [posts[0]]
    top = max(want, key=lambda p: p["likes"])

    def feed(q, f, s):
        return html(page("Dev feed", "<div id=feed></div><p id=more>Loading more…</p><script>let n = 0;"
                         "async function more() { const d = await (await fetch('/api/feed?from=' + n)).json(); n += d.length;"
                         "d.forEach(p => { const a = document.createElement('div'); a.className = 'post'; a.dataset.tag = p.tag;"
                         "a.innerHTML = `<h4>${p.t}</h4><span class=likes>${p.likes} likes</span> <span class=tag>#${p.tag}</span>`;"
                         "document.getElementById('feed').append(a); });"
                         "if (!d.length) document.getElementById('more').innerText = 'No more posts'; }"
                         "more(); addEventListener('scroll', () => { if (innerHeight + scrollY > document.body.scrollHeight - 200) more(); });</script>",
                         footer=False))

    def api(q, f, s):
        i = int(q.get("from", ["0"])[0])
        return jsonr(posts[i:i + 10])

    script = ("let last = -1;\n"
              "for (let i = 0; i < 40; i++) {\n"
              "  const n = document.querySelectorAll('.post').length;\n"
              "  if (n === last && document.getElementById('more').innerText.includes('No more')) break;\n"
              "  last = n; window.scrollTo(0, document.body.scrollHeight); await new Promise(r => setTimeout(r, 400));\n"
              "}\n"
              f"const posts = [...document.querySelectorAll('.post')].filter(p => p.dataset.tag === {json.dumps(tag)})\n"
              "  .map(p => ({title: p.querySelector('h4').innerText, likes: parseInt(p.querySelector('.likes').innerText)}));\n"
              "posts.sort((a, b) => b.likes - a.likes);\n"
              "({loaded: document.querySelectorAll('.post').length, matching: posts.length, top: posts[0]})")

    def solve(base):
        obs = yield Reply(plan("Open the feed", "Scroll to the end with a script, then pick the most-liked post with that tag", "Report it")
                          + f'\n<browse url="{base}/feed"/>',
                          thought="The feed loads more posts as you scroll. A script can keep scrolling until nothing new loads.")
        obs = yield "First I'll check how a post is built, so the script uses the real class names.\n" + inspect_js(posts[0]["t"])
        obs = yield ("Each post is a div.post with data-tag, an h4 title and .likes. #more says when the feed has ended. "
                     f"Script to scroll to the end and pick the top #{tag} post:\n<js>\n{script}\n</js>")
        d = script_json(obs) or {}
        top_ = d.get("top") or {}
        yield (f"After loading all {d.get('loaded', '?')} posts ({d.get('matching', '?')} tagged #{tag}), the most-liked #{tag} post is "
               f"\"{top_.get('title', '?')}\" with {top_.get('likes', '?')} likes.")

    return Task("infinite_feed", f"Go to {{base}}/feed and find the #{tag} post with the most likes. The feed loads more as you scroll, so check all of it.",
                {**base_routes(), "/feed": feed, "/api/feed": api}, solve,
                lambda s, ans: top["t"] in ans and str(top["likes"]) in ans)


# ---------------------------------------------------------------- I: <extract> then python for ranking
def leaderboard(rng: random.Random):
    players = rng.sample([f"{a} {b}" for a in FIRST for b in LAST], rng.randint(12, 25))
    rows = [{"p": p, "team": rng.choice(["Red", "Blue", "Green"]), "pts": rng.randint(100, 999), "games": rng.randint(5, 30)} for p in players]
    team = rng.choice(["Red", "Blue", "Green"])
    for r in rows[:3]:
        r["team"] = team          # at least three on the asked team
    rng.shuffle(rows)
    mine = sorted([r for r in rows if r["team"] == team], key=lambda r: -r["pts"] / r["games"])[:3]

    def board(q, f, s):
        t = ("<table class=board><tr><th>Player</th><th>Team</th><th>Points</th><th>Games</th></tr>" +
             "".join(f"<tr><td>{r['p']}</td><td>{r['team']}</td><td>{r['pts']}</td><td>{r['games']}</td></tr>" for r in rows) + "</table>")
        return html(page("Season leaderboard", t))

    def solve(base):
        obs = yield Reply(plan("Open the leaderboard", "Extract the table", f"Rank {team} players by points per game in python", "Report the top 3")
                          + f'\n<browse url="{base}/leaderboard"/>',
                          thought="Points per game needs maths: extract the rows, then compute in code, never in my head.")
        obs = yield '<extract query="leaderboard"/>'
        data = re.findall(r"([A-Z][a-z]+ [A-Z][a-z]+)\s*\|\s*(Red|Blue|Green)\s*\|\s*(\d+)\s*\|\s*(\d+)", obs)
        if not data:
            obs = yield ("<js>\n[...document.querySelectorAll('table.board tr')].slice(1).map(tr => [...tr.cells].map(c => c.innerText).join(' | ')).join('\\n')\n</js>")
            data = re.findall(r"([A-Z][a-z]+ [A-Z][a-z]+)\s*\|\s*(Red|Blue|Green)\s*\|\s*(\d+)\s*\|\s*(\d+)", obs)
        rows_py = ",\n".join(f'    ("{a}", "{b}", {c}, {d})' for a, b, c, d in data)
        code = (f"rows = [\n{rows_py}\n]\n"
                f"team = [r for r in rows if r[1] == {json.dumps(team)}]\n"
                "team.sort(key=lambda r: r[2] / r[3], reverse=True)\n"
                "for name, t, pts, games in team[:3]:\n"
                "    print(f\"{name}: {pts / games:.1f} points per game ({pts} in {games})\")")
        obs = yield f'<run lang="python">\n{code}\n</run>'
        lines = re.findall(r"^[A-Z][a-z]+ [A-Z][a-z]+: [\d.]+ points per game \(\d+ in \d+\)$", obs, re.M)
        yield f"Top 3 {team} players by points per game:\n" + "\n".join(f"{i}. {l.strip()}" for i, l in enumerate(lines[:3], 1))

    return Task("leaderboard", f"Open {{base}}/leaderboard and tell me the top 3 players on team {team} ranked by points per game.",
                {**base_routes(), "/leaderboard": board}, solve, lambda s, ans: all(r["p"] in ans for r in mine) and
                ans.index(mine[0]["p"]) < ans.index(mine[1]["p"]) < ans.index(mine[2]["p"]))


# ---------------------------------------------------------------- J: write and test your own page script
def js_tool(rng: random.Random):
    words = rng.sample(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"], 6)
    text = " ".join(rng.choice(words) for _ in range(rng.randint(60, 140)))
    w = rng.choice(words)
    count = text.split().count(w)

    def doc(q, f, s):
        return html(page("Transcript", f'<div id=doc><p>{text}</p></div><p class=meta>Words separated by spaces.</p>'))

    def solve(base):
        obs = yield Reply(f'<browse url="{base}/transcript"/>', thought="Counting words by eye is unreliable; count with a script.")
        obs = yield (f"<js>\nconst words = document.body.innerText.toLowerCase().split(/\\s+/).filter(Boolean);\n"
                     f"({{total: words.length, {w}: words.filter(x => x === {json.dumps(w)}).length}})\n</js>")
        d = script_json(obs) or {}
        yield f'The word "{w}" appears {d.get(w, "?")} times (out of {d.get("total", "?")} words).'

    return Task("js_count", f"Open {{base}}/transcript and count exactly how many times the word \"{w}\" appears on the page.",
                {**base_routes(), "/transcript": doc}, solve, lambda s, ans: re.search(rf"\b{count}\b", ans) is not None)


FAMILIES = {"shop_cheapest": shop_cheapest, "table_sum": table_sum, "paged_scrape": paged_scrape, "form_signup": form_signup,
            "delayed": delayed, "bulk_inbox": bulk_inbox, "api_stock": api_stock, "infinite_feed": infinite_feed,
            "leaderboard": leaderboard, "js_count": js_tool}
# How often each family is drawn: the script-heavy ones (what Harsh asked for) come up most.
WEIGHTS = {"shop_cheapest": 2, "table_sum": 3, "paged_scrape": 3, "form_signup": 2, "delayed": 1, "bulk_inbox": 3,
           "api_stock": 2, "infinite_feed": 3, "leaderboard": 2, "js_count": 2}


def route(task: Task, path: str):
    if path in task.routes:
        return task.routes[path]
    for k, fn in task.routes.items():
        if k.endswith("/*") and path.startswith(k[:-1]):
            return fn
    return None
