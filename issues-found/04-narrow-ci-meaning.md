# The decision-statistic CI is narrow because it only covers pool-accuracy uncertainty

The pair probability is computed exactly from the pool accuracy p: hypergeometric for v2/v3b, two Binomial(200, p) for v3a. Its 95% CI comes from bootstrapping the items, so it reflects uncertainty in p only (e.g. 62–64% for v3a). It is not the uncertainty you'd get from re-running whole experiments. The observed share of the 190 pairs (63%) is reported next to it and agrees. The verdict margin (25%) is far below every interval, so this doesn't affect the conclusion.
