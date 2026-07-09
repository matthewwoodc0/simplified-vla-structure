# 2026-07-09 - Research Console Dashboard

## Plain-English Summary

Added a separate, publishable web dashboard for the Simplified VLA experiment. It presents the current research verdict, evidence ladder, policy performance, hypothesis outcomes, suggested next tests, and selected video demos without changing simulator, controller, dataset, or training code.

The dashboard deliberately separates the scripted/task readiness result from the blocked learned-policy comparison. It reports the registered raw final as the current policy verdict and labels the H-EE-008 weighted-loss result as validation-only evidence.

## What To Review

- [ ] `dashboard/app/page.tsx`: dashboard content, evidence wording, interactive experiment filters, and demo selector.
- [ ] `dashboard/app/globals.css`: visual system and mobile layout.
- [ ] The visual-evidence section: confirm that the labels distinguish current evidence from historical clips.
- [ ] Suggested next step: registered final evaluation for the validated gripper-weighted loss.

## Implementation Details

The dashboard is a standalone Sites/Vinext project at `dashboard/`, keeping it isolated from the MuJoCo application. It is based on the registered Phase 5 evidence files and the current `researchnotes.md` verdicts.

The page includes four evidence layers, success-rate bars for raw final and weighted validation, raw-final strict-gate metrics, a filterable recent-experiment ledger, a hypothesis table, three local MP4 demos, and a decision queue. Demo captions explicitly state their scope.

## Evidence And Verification

```bash
cd dashboard && npm run build
```

- Result: passed; Vinext produced deployable output.
- What this proves: the dashboard compiles successfully.
- What it does not prove: that deployed hosting will automatically ingest future experiment artifacts. Current content is a curated evidence snapshot and should be refreshed when results change.

## Demo Videos / Visual Artifacts

- Open this video: [scripted pickup physics audit](</Users/matthewwoodcock/Documents/Simplified VLA Structure/outputs/pickup_showcase_physics_audit.mp4>).
- Open this video: [Phase 6a RGB dataset preview](</Users/matthewwoodcock/Documents/Simplified VLA Structure/outputs/phase6a_vision_sample/preview.mp4>).
- Open this video: [historical EE successful rollout](</Users/matthewwoodcock/Documents/Simplified VLA Structure/outputs/ee_bc_success.mp4>).

The site packages copies of these clips under `dashboard/public/demos/`. Regenerate the source clips using the existing render scripts before replacing the packaged copies.

## Decisions Made

- Decision: use a separate dashboard project.
  Reason: it preserves the existing simulation repository and uncommitted research work.
- Decision: show raw final and weighted validation separately.
  Reason: combining them would overstate Phase 5 readiness.
- Decision: include only three compact video artifacts.
  Reason: they provide useful visual evidence without bloating the dashboard unnecessarily.

## Risks And Limitations

- Risk: experiment values are currently a curated snapshot in the dashboard component.
  Why it matters: new runs require updating the dashboard data or adding an artifact-ingestion layer.
- Risk: current browser preview could not be opened in this environment because the local dev server was denied a required inspector port.
  Why it matters: the production build passed, but visual browser QA was not completed here.
- Risk: Sites version saving returned an upstream `Unknown tool` service error.
  Why it matters: the source was pushed to the Sites source repository, but a production version could not be saved or deployed in this session.

## Action Items

- [ ] Review dashboard terminology and evidence emphasis.
- [ ] Add a small artifact-export script or JSON API if automatic dashboard refresh after each experiment is desired.
- [ ] Retry Sites version save/deployment once the Sites service error is resolved.

## Files Changed

- `dashboard/` - standalone research-console website and packaged demo assets.
- `reports/2026-07-09-research-console-dashboard.md` - implementation audit and handoff.

## Current Verdict

**Needs review / deployment retry.** The dashboard builds and its source has been pushed to the new Sites source repository. Its production deployment is blocked by the Sites version-save service error, not by an application build failure.
