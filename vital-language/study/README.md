# Blind vitality rating study (Week 4)

Tests the central validation question (H3): does our MEASURED surprisal
multifractality correspond to HUMAN-felt vitality? If not, the metric is
decoration; if yes, the chaos>matched result has perceptual grounding.

## Pipeline

1. **Build** a blind study from run outputs (already done -> `manifest.json`,
   `key.json`). Rebuild / resize:
   ```
   python -m vitality.study.build_study outputs/longrun_*/longrun.jsonl \
       --out study --min-tokens 200 --max-items 12 --seed 0
   ```
   - `manifest.json` is blind (passages = {uid, text} only; condition hidden).
   - `key.json` maps uid -> condition + metrics (analysis only; do NOT show raters).
   - `--max-items` caps length for rater fatigue, balanced across categories.
     12 items ~ 18 min; omit for the full 35-item study.

2. **Rate**: open `study/rate.html` in any browser (no server needed). Each
   rater loads `manifest.json`, enters an id, rates every passage on 8
   dimensions (0-100) and marks the one that feels "most alive" per item, then
   downloads `ratings_<id>.json`. Conditions are never shown; passage order is
   pre-randomized per the manifest.

3. **Analyze**: collect the rater files and run
   ```
   python scripts/analyze_study.py study/key.json ratings_*.json
   ```
   Reports per-condition x dimension means, "most alive" win-rate, and the
   headline **correlation(surprisal_width, felt vitality)** with bootstrap CI.

## Recommended

5-20 raters (per spec). Even 3-5 gives a first read on H3. Hand each rater the
same `manifest.json`; keep `key.json` private until analysis.
