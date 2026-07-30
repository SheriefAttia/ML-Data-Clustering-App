# Next Steps

Working notes for continuing the deployment effort in a future session.
Written after a deploy attempt was **halted before any build or deploy ran** — see
"Why deployment stopped" below. No infrastructure was created, and no application
code was modified.

---

## 1. Current state of the repo

```
ML-Data-Clustering-App/
├── LICENSE
├── README.md
├── main.py            # entire application, 402 lines
├── next-steps.md      # this file
└── requirements.txt
```

- **Entry point:** `main.py` → `if __name__ == "__main__": app = Application(); app.mainloop()`
- **Architecture:** a single `Application(tk.Tk)` class (`main.py:28`) plus one helper
  transformer, `DateTransformer` (`main.py:18`). No packages, no modules, no tests.
- **Dependencies** (`requirements.txt`): pandas, numpy, scikit-learn, matplotlib,
  seaborn, xlrd, openpyxl — **plus `tkinter`**, which is implied by the code but not
  listed (it ships with CPython).

### Where the logic lives

| Concern | Location |
|---|---|
| UI layout (all widgets/buttons) | `Application.create_widgets` — `main.py:42-149` |
| File load (Excel) | `browse_file`, `load_data` — `main.py:161-172` |
| Dynamic column widgets after load | `update_gui_elements` — `main.py:174-188` |
| pandas wrangling (filter/group/pivot/sort/EDA) | `main.py:189-296` |
| **Core clustering pipeline** | **`cluster_data` — `main.py:312-361`** |
| Plotting (2D seaborn / 3D matplotlib) | `visualize_clusters` — `main.py:363-386` |
| Excel export | `save_clustered_data` — `main.py:388-397` |

`cluster_data` is the heart of the app: it splits columns by dtype, builds a
`ColumnTransformer` (numeric → `SimpleImputer` + `StandardScaler`; categorical →
`SimpleImputer` + `OneHotEncoder`; datetime → `DateTransformer`), then chains
`VarianceThreshold` → `TruncatedSVD` → the chosen estimator (`KMeans`, `DBSCAN`, or
`AgglomerativeClustering`) into one `sklearn.pipeline.Pipeline` and calls
`fit_predict`.

---

## 2. Why deployment stopped

Two independent blockers, both verified:

### Blocker A — no deploy target matches this repo

Detection was run against the repo contents. **Zero markers were found:**

| Target | Marker looked for | Present? |
|---|---|---|
| Vercel | `vercel.json` / `.vercel` | No |
| Cloudflare | `wrangler.toml` / `wrangler.jsonc` | No |
| Fly.io | `fly.toml` | No |
| Netlify | `netlify.toml` | No |
| Kubernetes | k8s manifests / Helm chart | No |
| Supabase | `supabase/config.toml` | No |

There is also no `Dockerfile`, no `package.json`, and no web framework — so there is
**no build artifact to produce and no URL to verify.**

The deeper issue is architectural: this is a **desktop GUI application**. It calls
`filedialog.askopenfilename` (`main.py:167`), `messagebox.*` throughout, and
`plt.show()` (`main.py:386`) — all of which require an interactive display and a
local filesystem. None of that survives a move to serverless or a headless
container without changes. Deploying it as-is would produce a clean CLI exit code
but nothing a user could actually use.

### Blocker B — no deploy credentials in the environment

Checked the environment for `FLY_API_TOKEN`, `VERCEL_TOKEN`,
`CLOUDFLARE_API_TOKEN`, `NETLIFY_AUTH_TOKEN`, `SUPABASE_ACCESS_TOKEN`, and
`KUBECONFIG`. **None are set.** No auth was improvised.

---

## 3. Decision required before any deploy work

Pick one direction. **Option 1 is the recommendation** — it is the only path that
yields a genuinely usable production URL that can be honestly verified.

### Option 1 — Port the UI to a web app, then deploy (recommended)

Streamlit (or Gradio) maps almost 1:1 onto the existing controls, and the sklearn
pipeline logic is reused untouched.

| Tkinter today | Streamlit equivalent |
|---|---|
| `filedialog.askopenfilename` | `st.file_uploader` |
| `num_clusters_spinbox` | `st.number_input` |
| `method_combobox`, `missing_values_combobox` | `st.selectbox` |
| `variance_threshold_entry`, `dbscan_eps_entry` | `st.number_input` / `st.slider` |
| `messagebox.showinfo/showerror` | `st.success` / `st.error` |
| `plt.show()` | `st.pyplot(fig)` |
| `save_clustered_data` + save dialog | `st.download_button` |

Work items:
1. Extract the pipeline logic out of the Tkinter class into a pure, testable
   function — e.g. `build_pipeline(config) -> Pipeline` and
   `run_clustering(df, config) -> (labels, svd_components, score)`. This is the
   critical refactor: today `cluster_data` reads widgets directly off `self`
   (`self.variance_threshold_entry.get()` etc.), so it cannot be tested or reused.
2. Add `app.py` (Streamlit UI) that calls those pure functions.
3. Keep `main.py` working as the desktop entry point by having it call the same
   pure functions — avoids forking the logic into two copies.
4. Add `streamlit` to `requirements.txt`.
5. Write `Dockerfile` (python:3.11-slim, `MPLBACKEND=Agg`, non-root user, bind
   `0.0.0.0:8080`) and `fly.toml`.
6. Add `FLY_API_TOKEN` to Settings → Secrets (see §4).
7. `flyctl deploy`, streaming output.
8. Verify by fetching the live URL and showing the actual HTTP response — not just
   the CLI exit code.

### Option 2 — Containerize the Tkinter app as-is (Xvfb + noVNC)

Preserves current code with no refactor, but delivers a clunky browser-VNC desktop
rather than a web app, and file dialogs still write to ephemeral container storage.
Deploy to Fly.io. Lower effort, noticeably worse result.

### Option 3 — Ship a desktop binary instead

PyInstaller build attached to a GitHub Release via `gh`. Not a "deployment" with a
URL, but it is the most honest fit for what this program actually is.

### Option 4 — Set up CI only

Lint + smoke-import on push. Cheapest, and valuable regardless of which option
above is chosen, since there are currently zero tests.

---

## 4. Credential needed (for Options 1 and 2)

Add in **Settings → Secrets**:

- **`FLY_API_TOKEN`** — generate with `flyctl tokens create deploy` from a machine
  already logged into Fly, or via the Fly.io dashboard under
  **Account → Access Tokens**.

If the target changes, the required secret changes accordingly
(`VERCEL_TOKEN`, `CLOUDFLARE_API_TOKEN`, `NETLIFY_AUTH_TOKEN`,
`SUPABASE_ACCESS_TOKEN`, or file-shaped `FILE__KUBECONFIG` for a cluster).

---

## 5. Pre-existing bugs found while reading the code

These are unrelated to deployment and were **not** introduced by any change here.
Worth fixing independently — ideally alongside the Option 1 refactor.

1. **`start_machine_learning` is defined twice** — `main.py:277` and `main.py:290`.
   Python keeps the *second* definition, so the earlier one (which showed a
   "not implemented yet" message box) is dead code and the wired-up button silently
   does nothing (`pass`). Either implement it or remove the button at `main.py:148`.

2. **Two buttons collide on the same grid cell** — "Generate Pivot Table"
   (`main.py:144`) and "Perform EDA" (`main.py:146`) are both placed at
   `column=0, row=16`. The second overlaps the first, so one is effectively
   unreachable. Renumber the rows.

3. **`generate_pivot_table` builds `index_columns` and `columns_column` from the
   same checkbox state** (`main.py:243-256`), so a pivot's index and columns are
   always identical — almost certainly not the intent. It also ignores the
   `pivot_index_var` / `pivot_columns_var` comboboxes and `get_pivot_setting`
   helper that exist for this purpose.

4. **Pivot output goes to stdout**, not the GUI — `print(pivot_table)`
   (`main.py:265`). Invisible to a user running the app as a window.

5. **`cluster_data` re-reads the Excel file from disk** (`main.py:313`), silently
   discarding any filtering, grouping, or sorting the user applied beforehand.

6. **`silhouette_score` will raise if a run produces a single cluster** — and for
   `DBSCAN` the score is skipped entirely (`main.py:364-366`), so DBSCAN users get
   no quality metric at all.

7. **No input validation on numeric entry fields** — `float(...)` / `int(...)` on
   raw widget text (`main.py:335-336`, `345-346`) raises `ValueError` on any
   non-numeric input. It is caught by the broad `except` in `start_clustering`
   (`main.py:304`), which surfaces a raw exception string instead of a useful
   message.

8. **`update_gui_elements` appends duplicate checkboxes on every load** — it never
   clears previously created widgets or `self.group_by_vars`, so loading a second
   file stacks stale controls on top of the old ones.

9. **`xlrd>=2.0.1` cannot read `.xlsx`** — as of 2.0, xlrd supports only legacy
   `.xls`. The file dialog filters for `*.xlsx` (`main.py:167`), which is handled by
   `openpyxl`. The `xlrd` pin and its comment in `requirements.txt` are misleading.

---

## 6. Suggested order of work next session

1. Confirm which option from §3 to pursue.
2. If Option 1 or 2: add `FLY_API_TOKEN` to Settings → Secrets **first** — no
   build should start before credentials are verified present.
3. Fix the §5 bugs that overlap the refactor (items 1, 2, 5, 7).
4. Extract pure `build_pipeline` / `run_clustering` functions; add tests for them
   (none exist today — this is the safety net for every later change).
5. Build the web UI, `Dockerfile`, and `fly.toml`.
6. Deploy, then verify by fetching the live URL and showing the real response.
