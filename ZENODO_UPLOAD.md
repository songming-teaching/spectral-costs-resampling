# Zenodo upload instructions

This document records how to mint a DOI for this archive, and what exactly goes in the
record. It is written so that the operation can be repeated for a later revision.

**Repository:** https://github.com/songming-teaching/spectral-costs-resampling
**Snapshot to archive:** commit `1b1e39a` (tag `v1.0-submission`)
**Size:** 140 files, 5.1 MB (excluding `.git`)

---

## 1. Which route to take

Two routes mint a DOI. They differ in who builds the archive file.

| | Route A — GitHub integration | Route B — manual upload |
|---|---|---|
| What Zenodo archives | its own `.zip` of the tagged commit | a `.zip` you build and upload |
| Trigger | you publish a GitHub **Release** | you click upload |
| Future revisions | push tag → new version DOI automatically | repeat by hand |
| Recommended | **yes**, for the `v1.0-submission` release | use if you need to attach extra files |

Both produce a **version DOI** (e.g. `10.5281/zenodo.1234567`) plus a **concept DOI**
(e.g. `10.5281/zenodo.1234566`) that always points at the newest version. **Cite the
version DOI in the paper; link the concept DOI from the README.**

---

## 2. Route A — GitHub → Zenodo integration

### 2.1 One-time setup

1. Go to <https://zenodo.org> and sign in **with GitHub** (the account that owns the repo,
   `songming-teaching`).
2. Top-right menu → **Settings** → **GitHub** tab.
3. Find `songming-teaching/spectral-costs-resampling` in the repository list.
4. Flip its switch **On**. (If the repo is not listed, click *Sync now*.)

From this moment, every GitHub Release you publish in that repo is archived automatically.

### 2.2 Mint the DOI

1. Create the tag and release. From a clone of the repository:

   ```bash
   git tag -a v1.0-submission -m "Archive for IEEE TSP submission"
   git push origin v1.0-submission
   gh release create v1.0-submission \
       --title "v1.0-submission" \
       --notes "Archival snapshot accompanying the IEEE TSP submission."
   ```

   Or use the GitHub web UI: *Releases* → *Draft a new release* → choose tag
   `v1.0-submission` → *Publish release*.

2. Within a minute or two Zenodo sends a confirmation email and the record appears under
   <https://zenodo.org/account/settings/github/> with a **DOI badge**.
3. Click the badge to open the record, then **Edit** it and fill in the metadata of
   section 4 below. The record is public immediately once the release is published, so
   prepare the metadata text beforehand.

---

## 3. Route B — manual upload

1. Build the snapshot zip from the tagged commit, so the archive provably matches a commit:

   ```bash
   cd spectral-costs-resampling
   git archive --format=zip --prefix=spectral-costs-resampling-v1.0.0/ \
       -o ../spectral-costs-resampling-v1.0.0.zip v1.0-submission
   ```

   This excludes `.git/` and honours `.gitattributes`. Expected result: about 5.1 MB,
   reproducing the 140 committed files.

   Alternative if you prefer the GitHub-generated zip:
   <https://github.com/songming-teaching/spectral-costs-resampling/archive/refs/tags/v1.0-submission.zip>
   (larger, because GitHub also packs `.gitattributes`-covered metadata, but equivalent).

2. Go to <https://zenodo.org/uploads/new> (or *New upload*).
3. **Drag the zip in** as the only file. Do not also upload the loose directory tree; one
   zip keeps the record unambiguous and matches the tag one-to-one.
4. Fill in the metadata of section 4.
5. Set **Visibility** to *Public* only when the metadata is final, then click **Publish**.

---

## 4. Metadata to enter

Copy these field by field. The `Description` block should be pasted as-is; Zenodo renders
Markdown.

**Resource type:** `Software`
(It bundles code, data and the manuscript PDFs. `Dataset` is defensible; `Software` matches
the fact that the repository is the primary artifact and `CITATION.cff` declares
`type: software`.)

**Title**

```
Spectral Costs of Multinomial and Residual Resampling in High-Dimensional Particle Ensembles: A Free-Probability Analysis — code and data
```

**Creators** (order as on the paper)

| # | Name | Affiliation |
|---|---|---|
| 1 | Song, Deming | Faculty of Applied Sciences, Macao Polytechnic University, Macao, China |
| 2 | Lam, Chi Kin | Faculty of Applied Sciences, Macao Polytechnic University, Macao, China |

Add ORCID iDs if you have them. The corresponding author's e-mail is
`p2514650@mpu.edu.mo` (Deming Song); `cklamsta@mpu.edu.mo` (Chi Kin Lam).

**Description**

```markdown
Archival snapshot of the code and data accompanying the manuscript "Spectral Costs of
Multinomial and Residual Resampling in High-Dimensional Particle Ensembles: A
Free-Probability Analysis" (Deming Song and Chi Kin Lam, Macao Polytechnic University),
submitted to IEEE Transactions on Signal Processing.

The archive contains:

- the numerical experiments referenced in the main text and supplementary material —
  E1 (count law to spectrum map), E2 (rank threshold, twelve-point sweep), E3 (equal ESS
  with different null spaces), E4 (residual randomization at the integer boundary, and
  the atom-free window), T1 (finite-sample moment identity), and the appendix items
  A1–A4 (scale checks, non-Gaussian control, two-point population spectrum, solver
  sensitivity);
- the figure-generation scripts and the twenty figure panels in four print-width presets;
- the compact result tables (CSV) and per-experiment run logs;
- the LaTeX sources and compiled PDFs of both the main text and the supplementary
  material.

Each experiment is self-contained under `experiments/<id>/` with `code/`, `data/`, `fig/`
and `logs/` subdirectories, and is seeded (`SEED = 20260915`) for bit-reproducibility.
All stored statistics were pre-registered against closed-form predictions; every
pre-registered quantity is reported against theory with its Monte Carlo standard error in
the experiment logs.

Note on naming: the experiment identifiers (E1–E4, T1, A1–A4) are development identifiers
and are not the manuscript's figure numbers. The mapping is given in README section 3.
Ten of the twenty figure panels were produced but not used in the submitted manuscript;
their status is itemised in README section 4.

Reproduce with Python 3, numpy>=2.0, scipy>=1.10, matplotlib>=3.7; see `requirements.txt`
and README section 5.

Requires: `experiments/common/code/stage0_solvers.py` (two independent implementations of
the fixed-point equation for the limiting spectral distribution) and
`experiments/common/code/plot_panels.py` (per-panel PDF renderer).
```

**Keywords** (Zenodo takes free text; paste one per line)

```
particle filtering
resampling
random matrix theory
free probability
limiting spectral distribution
effective sample size
multinomial resampling
residual resampling
reproducibility
```

**License**

> Zenodo accepts **one** licence per record.

Choose **Creative Commons Attribution 4.0 International (CC BY 4.0)** — the data, figures
and documentation are the bulk of the record and are covered by it. Then add this sentence
at the end of the `Description` so the code's licence is not lost:

```markdown
Licensing: code (the Python files) and the LaTeX sources are released under the MIT
License; data, figures and documentation under CC BY 4.0. The file
`paper/src/IEEEtran.cls` is a third-party file from IEEE under the LaTeX Project Public
License. See `LICENSE` and `LICENSE-data` in the archive.
```

**Version:** `1.0.0`
**Language:** English
**Publisher:** Zenodo
**Related identifiers**

| Relation | Identifier |
|---|---|
| `is supplement to` | the DOI of the IEEE TSP article — add once the paper has one |
| `is supplemented by` / `is identical to` | the engrXiv preprint DOI, once moderation completes |
| `is source of` | `https://github.com/songming-teaching/spectral-costs-resampling` |

**Funding / Grants:** add if the work was funded.

---

## 5. After the DOI is minted

1. **Take the DOI back into the archive.** In `CITATION.cff`, uncomment the `identifiers`
   block and fill in the version DOI:

   ```yaml
   identifiers:
     - type: doi
       value: "10.5281/zenodo.<NNNNNNN>"
       description: "Zenodo archive of this release"
   ```

   Add a DOI badge to the top of `README.md`, then commit and push:

   ```bash
   git add CITATION.cff README.md
   git commit -m "Record the Zenodo DOI for v1.0.0"
   git push
   ```

   Note that this new commit sits *after* the archived tag. That is expected: the DOI points
   at the tag, and the badge merely advertises it. Do **not** move the tag.

2. **Disclose both DOIs to the editorial office.** The cover letter to *IEEE Transactions
   on Signal Processing* already undertakes to communicate the **engrXiv** DOI once it is
   assigned. Add the **Zenodo** DOI in the same message, so that the disclosed record of
   prior or parallel dissemination is complete. The two are different artifacts — engrXiv
   carries the manuscript, Zenodo carries the code and data — and neither is a prior
   publication of the other.

3. **Keep the concept DOI in mind.** If you later issue a corrected release (for example
   after responding to reviewers), publish it as a new version of the *same* Zenodo record
   rather than a new record, and cite the new version DOI in the revision. The concept DOI
   keeps resolving to the latest.

---

## 6. Caveats

- **Public means public.** Publishing the Zenodo record and the GitHub release exposes the
  manuscript PDFs and every result table immediately. Confirm you want this before
  submission review is complete. If you would rather not, keep the GitHub repo private — but
  note that Zenodo's GitHub integration **cannot** archive private repositories, so you
  would have to use Route B with a manually built zip.
- **The archive is a snapshot, not a living repository.** Nothing in it will change. Any
  subsequent edit lives in git and, if archived, in a new version.
- **Ten unused figure panels** are included deliberately (README section 4). They are
  pre-registered outputs, not errors. If you would prefer a strictly submission-matching
  record, delete `fig/w1.66_col2/` and the panels listed as unused, and rebuild.
- **Do not rename the experiment directories** to match the paper's figure numbers at this
  stage. The paths are recorded in the commit, the CSVs, the logs and this document; a
  rename would invalidate README section 3 and every cross-reference.
