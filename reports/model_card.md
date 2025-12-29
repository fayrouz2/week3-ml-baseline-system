 #Model Card — Week 3 Baseline

 ## Problem
 - Predict: <target> for <unit of analysis>
 - Decision enabled: <what action changes?>
 - Constraints: CPU-only; offline-first; batch inference

 ## Data (contract)
 - Feature table: data/processed/features.<csv|parquet>
 - Unit of analysis: <one row per --.>
 - Target column: <name>, positive class: <--.> (if binary)
 - Optional IDs (passthrough): <list>

 ## splits (draft for now)
 - Holdout strategy: random stratified (default) / time / group
 - Leakage risks: <what could leak?>

 ## Metrics (draft for now)
 - Primary: <metric> (why it matches the decision)
 - Baseline: dummy model must be reported

 ## Shipping
 - Artifacts: model + schema + metrics + holdout tables + env snapshot
 - Known limitations: <where will it fail?>
 - Monitoring sketch: <what would you watch?>