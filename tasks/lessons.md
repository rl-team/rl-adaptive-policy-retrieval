# Lessons Learned

## Git Practices

### No Co-Authored-By trailers in commits
When committing code, do NOT include `Co-Authored-By: Claude` or similar AI co-author trailers. Keep only the configured git user (rshar / Ruslan Sharifullin) as the commit author. If a co-author trailer slips in, amend the commit and force-push with `--force-with-lease`.

### Binary files should not be tracked in git
Large binary artifacts (.pkl, .npy, etc.) should NOT be committed to git, even for a course project. They are uncompressible, bloat the repository permanently, and are reproducible from scripts with known seeds. Instead:
- Add them to `.gitignore`
- Document the regeneration command in README or commit message
- Consider Git LFS only if regeneration is impractical

### Commit messages reference task IDs
Format: `R38-R45: evaluation pipeline, OPE, significance tests, and figures`. Include task IDs (R38, M31, H12, etc.) so teammates can trace commits to the implementation plan.

### Branch naming convention
Use `<initials>/<task-description>` format: `rs/r38-r45-eval-analysis`, `mg/offline_dataset`, `rs/fix-m29-m32-cleanup`.

## LaTeX Practices

### TikZ over raster images
When a diagram can be implemented as TikZ code in LaTeX, prefer that over raster images (.png). TikZ diagrams scale cleanly, are editable in the .tex source, and produce vector output. Reserve raster images for data-generated plots (matplotlib output).

### No baked-in figure captions in images
Never hardcode figure/table numbers or captions into matplotlib-generated images (e.g., "Figure 1. ..."). Figure numbering is controlled by LaTeX `\caption{}` and `\label{}`. Baked captions will be wrong if sections are reordered or figures are added.

### Publication-quality matplotlib defaults
Use `scripts/plot_style.py` → `apply_publication_style()` for consistent font sizes, DPI, grid alpha across all figures. Import from shared module instead of copy-pasting rcParams.

## Code Quality

### Dead code should be removed, not left
Remove unused functions, imports, and variables. Don't leave dead code "for later" — it creates confusion and maintenance burden. If code might be useful, reference it in a comment with the commit hash where it was removed.

### Avoid cross-script imports from CLI entrypoints
Don't import utility functions from CLI scripts (`scripts/evaluate_agent.py`). Instead, put shared functions in library modules (`evaluation/` package). CLI scripts should be thin wrappers around library code.

### Action index convention
The RL environment uses **relative action indices** (0-9 for retrieval, 10 for stop). The `BaselinePolicy` interface uses **absolute corpus indices**. Always be explicit about which convention a function expects. The BC bug (IndexError with corpus index 158 into 11-dim logits) was caused by this mismatch.

### Seed consistency
Project seeds: train=42, test=99, evaluation=42. All scripts should use these defaults. Document seed values in docstrings and commit messages.

## Prompt/Workflow Preferences

### Always read before editing
Never propose changes to code you haven't read. Read the file first. Understand existing patterns before modifying.

### Write observations for every task
After completing any task, write observations to the main-doc. Include: what was done, commands run, data obtained, analysis, conclusions, and academic implications. Use mermaid diagrams when they help explain concepts.

### No time estimates
Never give time estimates. Focus on what needs to be done, not how long it takes.

### Verify before marking complete
Run scripts, check outputs, verify data before marking a task complete. A task without demonstrated results is not complete.
