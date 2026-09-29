# A03 — Collaborative Filtering on MovieLens-100K

A browser application that predicts how a person would rate a film they have
never seen, using three different methods, and shows the reasoning behind every
prediction on screen.

Everything runs in the browser. No server-side code, no framework, no build
step, nothing to install.

**Contents:** [1. The problem](#1-the-problem) · [2. How it was solved](#2-how-it-was-solved) · [3. Bugs found](#3-bugs-found) · [4. The web app, part by part](#4-the-web-app-part-by-part) · [5. How to run it](#5-how-to-run-it) · [6. File map](#6-file-map)

---

# 1. The problem

## 1.1 The setup

Picture one enormous spreadsheet. Rows are people — 943 of them. Columns are
films — 1,682. A cell holds the rating that person gave that film, 1 to 5.

Almost the entire sheet is blank. Only **100,000 of the 1,586,126 cells** are
filled in, which is **6.3%**. The job is to guess what belongs in a blank cell:
*would user 1 enjoy Star Wars?*

The Week-2 assignment answered that by looking at the film itself — its genres.
Collaborative filtering throws all of that away. It never looks at what a film
*is*. It only looks at the pattern of who rated what. The entire idea is one
sentence:

> **People who agreed with you in the past will probably agree with you again.**

Everything below is bookkeeping on top of that sentence.

## 1.2 First difficulty — which direction do you compare?

That sentence can be read two ways, and they give different algorithms.

**User-based CF** compares *columns* of the matrix. Find people whose ratings
look like yours, then see what they thought of Star Wars.
*"People like you liked this."*

**Item-based CF** compares *rows*. Find films that get rated the way Star Wars
does, then see what you thought of those.
*"You liked these similar films."*

Same data, same arithmetic, rotated ninety degrees. The lecture presents this as
the headline decision. One finding here is that it is the *smaller* of the two
decisions.

## 1.3 Second difficulty — the holes, and why they are vicious

To compare two people you need a number for *how alike are they*. The standard
tool is **cosine similarity**: treat each person's ratings as a vector and
measure the angle between them. Small angle, similar taste.

But you can only compare on films that **both** people rated, and in a sheet
that is 93.7% empty, two random people often share a handful of films.
Sometimes exactly one.

Here is the poison. Two users share exactly one film. You gave it a **5**. They
gave it a **1**. You could not disagree more strongly. What does cosine say?

```
cosine = (5 × 1) / (5 × 1) = 1.0
```

**Perfectly identical.** And that is not a quirk of those two numbers — do the
algebra with any pair of values and the answer is always exactly 1.0. In one
dimension every vector points the same way. There is no angle left to measure.

So the pairs you know *least* about score *highest*, and they float to the top
of the neighbour list. The ranking is upside down precisely where it matters
most. Measured on the real data: **5.9% of the neighbours the system actually
chose scored exactly 1.0**, and predictions resting on five or fewer shared
films had error 1.0085 against 0.9250 for the rest.

There is a second, quieter version of the same disease. Every rating is
positive — nobody can rate below 1 — so every vector points into the same
corner of the space and *everybody* looks similar to *everybody*. Average
similarity across user pairs is **0.89**, with one pair in seven above 0.99.
The measure has almost no room left to tell people apart.

---

# 2. How it was solved

## 2.1 The predictor

Once you have neighbours, you do **not** simply average their ratings. You
centre each rating on that neighbour's own average first:

```
prediction = your average
           + weighted average of (neighbour's rating − neighbour's average)
```

Why it matters: a **4** from someone who averages 2.0 is enthusiasm. A **4**
from someone who averages 4.5 is a shrug. Plain averaging treats those as the
same evidence. Centring keeps the difference.

Written out, with the two axes swapped:

```
user-based:  pred(u,i) = mu_u + Σ_v sim(u,v)·(r_vi − mu_v) / Σ_v |sim(u,v)|
item-based:  pred(u,i) = mu_i + Σ_j sim(i,j)·(r_uj − mu_j) / Σ_j |sim(i,j)|
```

## 2.2 The three missing-value strategies

Each is one row of the lecture's table, and each is selectable live in the UI.

**`rated` — ignore the holes.**
Use only the films both people rated. Unshared cells leave the calculation
entirely. Simplest possible thing, and fully exposed to both problems above.

**`weighted` — distrust thin evidence.**
Compute the same similarity, then multiply it by a penalty based on how many
films the pair actually shares: `min(shared, 25) / 25`. Two shared films means
the score counts for 8%; twenty-five or more means full weight. This is a
direct, surgical fix, and it works completely — afterwards **not one**
prediction in 2,000 had a thin-evidence neighbour at the top of its list,
against 677 before.

**`mean` — fill the holes with each person's average.**
Pretend every blank cell holds that person's own average rating, then measure
everyone relative to their own average. This is Pearson correlation computed
over the union of the two rating sets.

## 2.3 Matrix factorization — a different kind of answer

Instead of deciding what a hole *means*, learn a compact description of every
user and every film — 10 numbers each — tuned so that multiplying a user's ten
numbers by a film's ten numbers reproduces the ratings you do have. Every blank
cell then has a value automatically.

Two variants are implemented:

- `plain` — the Week-3 specification exactly as written: `pred = p_u · q_i`
- `biased` — `pred = mu + b_u + b_i + p_u · q_i`, with L2 regularisation

## 2.4 Results

Held out 10% of the ratings, trained on the other 90%, repeated on three
different random splits. Lower RMSE is better.

| Method | RMSE |
|---|---|
| predict the global average | 1.1257 |
| predict the film's average | 1.0307 |
| user-based CF, `rated` | 0.9598 |
| user-based CF, `weighted` | 0.9518 |
| user-based CF, `mean` | 0.9327 |
| item-based CF, `rated` | 0.9455 |
| item-based CF, `weighted` | 0.9401 |
| **item-based CF, `mean`** | **0.9067** |
| matrix factorization, best of three configs | 0.9444 |

### Why mean imputation won

Subtracting each person's average does **two** things at once.

*It rescales the measure so it can express disagreement.* Once you subtract the
mean, some values go negative, so similarity can go negative too:

| strategy | average similarity | range |
|---|---|---|
| `rated` | 0.89 | 0 → 1.00 |
| `weighted` | 0.38 | 0 → 0.99 |
| `mean` | 0.02 | **−0.29** → 0.42 |

Only the last can say *"these two people actively disagree."* The first barely
distinguishes anybody from anybody.

*And it fixes thin evidence as a side effect.* If two people share one film out
of 200 each, that film adds one small term to the top of the fraction, while
the bottom counts all 200 films on each side. One over a big number is near
zero. The score collapses on its own, with no special penalty rule needed.

So `weighted` fixes one problem exactly; `mean` fixes both approximately, and
that turns out to be the better trade. **Being targeted is not the same as
being better.**

### Two findings that contradict the lecture

**The efficiency rule of thumb measures a different cost than it sounds like.**
The slide says item-based is more efficient when there are more users than
items. Measured per prediction:

- user-based compares you against everyone who rated the film (~54 people),
  walking ~95 entries each
- item-based compares the film against every film you rated (~95 films),
  walking ~54 entries each

The same two numbers multiplied in the other order — **15,765 versus 15,118
operations**. Identical. The cost is symmetric regardless of which side of the
matrix is larger. The rule only holds if you precompute the *whole* similarity
table, and even there the 3.11× difference in pair count shrinks to 1.2× in
real time.

**Cold start is two different failures, not one.**

| ratings the film has | error | does it answer? |
|---|---|---|
| 0 | — | **no** |
| 1 | 1.4571 | yes, confidently |
| 2–5 | 1.0446 | yes |
| 100+ | 0.9259 | yes |

A film with **zero** ratings is not hard, it is *impossible* — you are
averaging over neighbours that do not exist. No amount of clever similarity
reaches it. The honest output is "I don't know", which is what the code
returns.

A film with **one** rating is arguably worse. The system answers every time,
with no hesitation, and is 57% less accurate. **Confidence does not decay as
evidence thins.** It stays at 100% while accuracy falls apart, and nothing on
screen tells you to distrust it.

---

# 3. Bugs found

## 3.1 The `support` bug — mine, and the important one

**Symptom.** The first Top-5 list in item-based mode was:

```
1. Great Day in Harlem, A (1994)              — 5.000  (support 30,  1 rating)
2. Prefontaine (1997)                         — 5.000  (support 30,  3 ratings)
3. Marlene Dietrich: Shadow and Light (1996)  — 5.000  (support 30,  1 rating)
```

Five films tied at exactly 5.000, three of them with one or two ratings in the
entire dataset, each claiming to rest on "support 30". Across 100 users the
recommended films averaged **10.6** ratings against a catalogue average of
**54.1** — the recommender was systematically digging up the most obscure thing
it could find.

**Root cause.** The Top-5 filter dropped films with too little evidence, using a
field called `support`. In **user-based** mode that field counts neighbours, and
each neighbour *is* one real rating of the target film — so it genuinely
measures evidence. In **item-based** mode the neighbours are films *the user*
rated, so the same field was only counting the length of the user's own
history. For any active user it read exactly 30, every time, no matter how
obscure the target film was.

**Why it hid.** Nothing about the prediction maths was wrong. The number 30 is
perfectly plausible. Only the *guard* was measuring the wrong quantity — and a
guard that always passes looks exactly like a guard that is working.

**Fix.** `support` is now defined as *"how many people actually rated this
film"*, computed differently per mode so that it means the same thing in both:
the neighbour count for user-based, the film's own rater count for item-based.

**Verification.** Three checks:

- the one- and two-rating films vanished from the list, replaced by
  *A Close Shave* (103 ratings) and *The Third Man* (68)
- average popularity of item-based recommendations rose from **10.6 to 57.9**,
  now sitting on the catalogue average rather than an order of magnitude below
- the reported support for *Great Day in Harlem* changed from 30 to **1** — the
  number it always should have been

## 3.2 A claim of mine that the data killed

Not a code bug, but worth recording. I originally reported that significance
weighting improved item-based accuracy. Re-running the whole comparison on two
further random splits showed that gap is **0.002 on average and reverses
direction on one seed** — pure noise. The claim was removed and replaced with a
note that the demonstrable effect of `weighted` is on the *similarity
distribution*, not on error.

This is why the accuracy table is measured on three splits rather than one.

## 3.3 TensorFlow.js native backend is broken on these versions

Trying to speed up the matrix-factorization runs with `@tensorflow/tfjs-node`
fails immediately:

```
TypeError: (0 , util_1.isNullOrUndefined) is not a function
    at createTensorsTypeOpAttr (nodejs_kernel_backend.js:675:38)
```

`tfjs-node` calls `util.isNullOrUndefined` from `tfjs-core`, which no longer
exports it. Reproduced on **4.20.0 and 4.22.0**, in a clean install with no
other packages. Not our bug, and there is no workaround short of patching
`node_modules`, so all MF numbers come from the pure-JavaScript CPU backend —
roughly 160 seconds per epoch.

## 3.4 A bug in the test harness itself

`browser_check.js` loads the app's scripts into a sandbox and then inspects
them. The first version read values as properties of the sandbox object and got
`undefined`.

Cause: `let` at the top level of a script goes into V8's **global lexical
scope**, which later scripts in the same context can read but which never
becomes a property of the global object. Browsers apply exactly the same rule.
So `data.js`'s `ratings` *is* visible to `script.js` — just not as
`sandbox.ratings`. Fixed by evaluating expressions inside the context instead.

Worth recording because the app was correct the whole time; the *test* was
wrong.

## 3.5 Inherited: the Week-2 genre parser

Checked while deciding what to reuse from Week 2. That submission's
`data.js:56` still reads `fields.slice(5, 24)` — nineteen values against an
eighteen-name list, so every genre label is shifted by one position. The Week-2
report documented the fix (`slice(6, 24)`) but it never reached the shipped
file.

**Not used here** — this project needs only `u.data` and the film titles, never
the genre flags. Recorded so that anyone reusing that file knows to correct it
first.

---

# 4. The web app, part by part

**4 regions on screen, 7 controls.**

## 4.1 The four regions

| Region | Purpose |
|---|---|
| **1. Algorithm** | Choose *which method* computes the answer. Changes nothing until you predict. |
| **2. Predict a rating** | Choose *who* and *which film*, then run it. Results appear here. |
| **3. Matrix factorization** | Train the neural model. Only needed if you chose MF above. |
| **Status bar** (bottom) | Loading progress, and warnings such as in-sample predictions. |

## 4.2 The seven controls

### Region 1 — Algorithm

**① Approach** — three options:

- *User-based CF — compare users (columns)*: "people like you liked this"
- *Item-based CF — compare items (rows)*: "you liked similar films" — **default**
- *Matrix factorization — TensorFlow.js*: the learned model

**② Missing-value strategy** — three options. **Disappears under MF**, because
matrix factorization does not use one:

- *Rated items only* — ignore blank cells
- *Mean imputation (Pearson)* — treat blanks as that person's average
- *Weighted by common ratings* — penalise thin evidence — **default**

**③ Neighbourhood size k** — slider, 1 to 200, default 30. How many neighbours
vote on the answer. Also hidden under MF. The number beside the label updates
as you drag.

Underneath, a grey hint line rewrites itself to describe whichever approach is
selected.

### Region 2 — Predict a rating

**④ User** — 943 entries, each labelled `User 1 (272 ratings, mean 3.61)`, so
you can deliberately pick a heavy rater or a sparse one.

**⑤ Movie** — 1,682 entries, alphabetical, each labelled
`Star Wars (1977) — 583 ratings`. **The rating count is the useful part** — it
lets you choose a famous film or a nearly-unrated one on purpose.

**⑥ Predict rating** — one user, one film, one number, plus the full neighbour
table.

**⑦ Top-5 for this user** — ignores the film dropdown. Scores *every* film that
user has not rated and returns the best five. Takes a second or two: it is
running about 1,600 predictions.

### Region 3 — Matrix factorization

**Train MF model** — builds and fits the TensorFlow.js model, logging loss per
epoch. Slow, and the page freezes between epochs.

## 4.3 Reading the output

```
User 1 → Star Wars (1977):  4.80 / 5          ← the answer, colour-coded
30 of 271 candidate neighbours used · anchored on the item mean 4.358
· support: 583 · raw value before clamping: 4.804
┌─────────────────┬────────────┬────────────┬────────┬─────────┐
│ Top neighbour   │ similarity │ co-ratings │ rating │ centred │
└─────────────────┴────────────┴────────────┴────────┴─────────┘
```

| Field | What it means |
|---|---|
| **candidate neighbours** | how many were available, and how many actually voted |
| **anchored on** | the baseline the prediction adjusts away from — the user's average (user-based) or the film's average (item-based) |
| **support** | how many people really rated this film. **The trust indicator.** Low support means the answer is a guess wearing a confident face |
| **raw value** | the number before clamping into 1–5. Far outside that range means the neighbours disagreed violently |
| **co-ratings** (column) | how many films each neighbour actually shares with you. **Watch this one.** Similarity 1.0000 with co-ratings 1 is the pathology from §1.3 |
| **centred** (column) | that neighbour's rating minus their own average — the quantity actually being averaged |

---

# 5. How to run it

## 5.1 Start the app

```sh
cd submit
python3 -m http.server 8080
```

Then open **http://localhost:8080**

A web server is required. Opening `index.html` directly as a `file://` URL will
**not** work, because browsers block `fetch` on local files and the app cannot
read `u.data`.

Wait for the status bar to read *"Ready — 100,000 ratings, 943 users, 1,682
rated films."* The buttons then enable themselves. **The two CF modes need no
training** and work immediately.

## 5.2 Six things worth trying

**1 — See the working behind a prediction.** Leave the defaults, pick any user
and film, press **Predict rating**. The neighbour table underneath is the whole
algorithm made visible.

**2 — Watch the strategies disagree.** Keep user and film fixed and switch the
strategy dropdown through `rated` → `weighted` → `mean`, predicting each time.
Look at the **similarity column**, not just the final number. Under `rated` the
values sit at 0.95–1.00 and barely differ; under `mean` they spread out and
some go negative. That is §1.3 happening live, and it is why `mean` wins.

**3 — Reproduce the report's worked example.** User 1, *Star Wars (1977)*,
**User-based**, strategy `rated`, k = 30 → **4.28**. The top neighbour should be
user 516 at **0.992596** on 10 co-rated films. That number was hand-computed as
182 / (√205 × √164) and re-derived independently in `verify.js`.

**4 — Break it on purpose.** The film dropdown shows each title's rating count.
Find one showing **1 rating** — for instance *Great Day in Harlem, A (1994)* —
and predict in item-based mode. You get a confident **5.00** with `support: 1`.
One observation behind a maximally confident answer: the failure that §2.4
argues is more dangerous than an outright refusal.

**5 — See the fixed bug.** Press **Top-5 for this user** in item-based mode.
You get sensible films with real support counts. Before §3.1 was fixed, that
list was three films with one rating each, all at 5.000, every one claiming
"support 30".

**6 — Drag k to 5, then to 200.** Predictions swing at the extremes and settle
in the middle.

## 5.3 One caveat when interpreting the app

The app builds its index from **all 100,000 ratings**. If you predict a film the
user has already rated, that rating was part of the input — so the result is a
fit, not a forecast. The app says so in italics under the result.

Every accuracy figure in the report comes from the held-out split computed by
`eval.js`, never from the app.

## 5.4 Run the checks

```sh
node test.js            # 30 unit tests, every answer hand-computed on a 3x3 matrix
node verify.js 1 50 30  # re-derives one real prediction independently of cf.js
node browser_check.js   # 20 checks on the app wiring against a stub DOM
node eval.js all        # regenerates every table in section 4 of the report
```

Tested on Node v23.11.0. No dependencies.

The matrix-factorization arm needs TensorFlow.js, which the browser gets from
the CDN and Node does not:

```sh
npm install @tensorflow/tfjs@4.22.0
node eval_mf.js all     # slow: ~160 seconds per epoch on the pure-JS backend
```

Do **not** install `@tensorflow/tfjs-node` to speed this up — see §3.3.

---

# 6. File map

| File | Role |
|---|---|
| `index.html` | Structure only — three panels, the controls, the script tags |
| `style.css` | Appearance only |
| `data.js` | Loads and parses `u.item` / `u.data`. Nothing else touches the file format |
| `cf.js` | **All neighbourhood scoring** — index, similarities, both predictors, Top-N |
| `mf.js` | Matrix factorization model (TensorFlow.js) |
| `script.js` | UI glue — **computes nothing** |
| `eval.js` | Offline evaluation on a seeded 90/10 split |
| `eval_mf.js` | Matrix-factorization sweep on the same split |
| `test.js` | Unit tests |
| `verify.js` | Independent re-derivation of one prediction |
| `browser_check.js` | Headless smoke test of the app wiring |
| `u.data`, `u.item` | MovieLens-100K, byte-identical to the Week-2 copies |
| `RESULTS.md` | Raw output of `node eval.js all` — the source of every table in §4 of the report |
| `mf_results.jsonl` | One line per trained MF model; §4-F of the report is generated from this file, not retyped from it |
| `mf_run.log`, `mf_run2.log` | stdout of the two `eval_mf.js` runs, including the per-epoch losses quoted in the report |

## The design rule that shapes all of it

**`script.js` computes nothing.** Not one similarity, not one average. It reads
the dropdowns, calls a function, formats the result.

That sounds like ordinary tidiness, but it is doing specific work here. Because
`cf.js` and `mf.js` contain no DOM code and no `fetch`, the identical files load
two ways:

- in the browser via `<script>` — they publish `window.cf` / `window.mf`
- in Node via `require` — they publish `module.exports`

So the offline harness that produced every number in the report runs **the same
file the page runs**. Nothing is re-implemented for measurement. That is the
mistake that cost accuracy in the Week-2 write-up, where the documented numbers
came from a re-implementation rather than from the application itself.

## Other design notes

**Abstention.** `predictUserBased` and `predictItemBased` return `null` when no
neighbour with positive similarity exists, rather than quietly substituting a
mean. The caller decides what to do; `eval.js` reports coverage as its own
column. Without this, the 0-rating films in §2.4 would have shown a mediocre
error instead of an honest 0% coverage.

**Clamping.** Predictions are clamped to 1–5 only at the very end. The
unclamped value stays available and is displayed, because a raw value far
outside the range is a signal that the neighbours disagreed.
