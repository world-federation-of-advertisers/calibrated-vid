# Two-EDP Reference-ID-anchored VID walkthrough

Both EDPs run the same static ranked-pool VID model. The model does not change day by day. The TEE coordinator changes only the rank assigned to each newly observed account, and every account keeps its first rank forever.

The 5-billion-value Reference-ID namespace is a join signal, not the VID pool. A matching Reference ID causes two EDP accounts to receive the same local rank. Proprietary-only accounts can also share a rank, but only as an aggregate calibration allocation; that sharing does not assert that the two proprietary IDs identify the same real person.

| Day | EDP | Account | Reference ID | Rank / VID | Why |
|---:|---|---|---:|---|---|
| 1 | EDP_A | `a-alice` | 101 | rank 0 → `VID(10000000+Feistel(0))` | new reference anchor |
| 1 | EDP_A | `a-bob` | 102 | rank 1 → `VID(10000000+Feistel(1))` | new reference anchor |
| 1 | EDP_A | `a-carol` | — | rank 2 → `VID(10000000+Feistel(2))` | new shared fallback rank |
| 1 | EDP_A | `a-dan` | — | rank 3 → `VID(10000000+Feistel(3))` | new edp exclusive rank |
| 1 | EDP_B | `b-alice` | 101 | rank 0 → `VID(10000000+Feistel(0))` | matched existing reference anchor |
| 1 | EDP_B | `b-carol` | — | rank 2 → `VID(10000000+Feistel(2))` | new shared fallback rank |
| 1 | EDP_B | `b-erin` | — | rank 4 → `VID(10000000+Feistel(4))` | new edp exclusive rank |
| 2 | EDP_A | `a-alice` | 101 | rank 0 → `VID(10000000+Feistel(0))` | reused frozen account mapping |
| 2 | EDP_A | `a-carol` | — | rank 2 → `VID(10000000+Feistel(2))` | reused frozen account mapping |
| 2 | EDP_A | `a-eve` | — | rank 5 → `VID(10000000+Feistel(5))` | filled opposite edp only rank |
| 2 | EDP_B | `b-alice` | 101 | rank 0 → `VID(10000000+Feistel(0))` | reused frozen account mapping |
| 2 | EDP_B | `b-bob` | 102 | rank 1 → `VID(10000000+Feistel(1))` | matched existing reference anchor |
| 2 | EDP_B | `b-eve` | 303 | rank 5 → `VID(10000000+Feistel(5))` | new reference anchor |
| 3 | EDP_A | `a-eve` | 303 | rank 5 → `VID(10000000+Feistel(5))` | reused frozen account mapping |
| 3 | EDP_A | `a-frank` | — | rank 6 → `VID(10000000+Feistel(6))` | new edp exclusive rank |
| 3 | EDP_B | `b-eve` | 303 | rank 5 → `VID(10000000+Feistel(5))` | reused frozen account mapping |
| 3 | EDP_B | `b-grace` | — | rank 7 → `VID(10000000+Feistel(7))` | new edp exclusive rank |
| 4 | EDP_A | `a-heidi` | — | rank 9 → `VID(10000000+Feistel(9))` | new edp exclusive rank |
| 4 | EDP_B | `b-heidi` | 404 | rank 8 → `VID(10000000+Feistel(8))` | new reference anchor |
| 5 | EDP_A | `a-heidi` | 404 | rank 9 → `VID(10000000+Feistel(9))` | reused frozen account mapping; **late anchor conflict flagged** |
| 5 | EDP_B | `b-ivan` | — | rank 9 → `VID(10000000+Feistel(9))` | filled opposite edp only rank |

## Cumulative report state

| Day | Requested A∩B | Achieved A∩B | Reach A | Reach B | Union | Direct anchors | Synthetic fallback | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2 | 2 | 4 | 3 | 5 | 1 | 1 | EXACT |
| 2 | 4 | 4 | 5 | 5 | 6 | 2 | 2 | EXACT |
| 3 | 3 | 4 | 6 | 6 | 8 | 3 | 1 | PROJECTED_UP_TO_IMMUTABLE_LOWER_BOUND |
| 4 | 4 | 4 | 7 | 7 | 10 | 3 | 1 | EXACT |
| 5 | 5 | 5 | 7 | 8 | 10 | 3 | 2 | EXACT |

## What the difficult cases show

- **Different email coverage works.** On day 2, EDP_B has Eve's email and EDP_A does not. The Reference-ID rank is fixed at EDP_B, and the residual allocator may place EDP_A's new proprietary account on that already occupied rank when the calibrated overlap target supports it.
- **Past reports are not loaded.** Stability comes from the persistent account-to-rank and Reference-ID-to-rank maps. Re-running any old impressions produces the same VIDs.
- **A lower later target cannot erase old overlap.** Day 3 asks for three shared people after four have already been committed. The output remains four and is flagged as a projection.
- **Late identity evidence can conflict with frozen labels.** Day 5 reveals an email after Heidi's EDP_A account was already assigned elsewhere. The allocator preserves the old VID, flags the missed anchor, and can compensate only through other new assignments. This is a real accuracy limit of immutable online labeling, not a reporting inconsistency.
- **Single-publisher reach is protected.** Within each EDP, no two distinct accounts are put on the same rank. Cross-EDP sharing changes deduplication, not either publisher's reach.

## Email-coverage feasibility under the strict design

This table assumes reach A = 600,000, reach B = 500,000, desired overlap = 300,000, and 60% conditional agreement when both EDPs have email. Only no-email accounts are allowed to supply synthetic overlap. The result is a capacity check, not an accuracy claim.

| Email coverage A | Email coverage B | Direct Reference-ID overlap | Flexible A | Flexible B | Maximum reachable overlap | Gap to target |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 10% | 1,800 | 540,000 | 450,000 | 500,000 | 0 |
| 10% | 90% | 16,200 | 540,000 | 50,000 | 500,000 | 0 |
| 50% | 50% | 45,000 | 300,000 | 250,000 | 500,000 | 0 |
| 90% | 10% | 16,200 | 60,000 | 450,000 | 500,000 | 0 |
| 90% | 90% | 145,800 | 60,000 | 50,000 | 255,800 | 44,200 |
| 95% | 95% | 162,450 | 30,000 | 25,000 | 217,450 | 82,550 |

The asymmetric 10%/90% cases remain feasible because the low-coverage EDP supplies a large flexible residual. The 90%/90% case can fail when conditional email agreement is only 60%: too many unmatched email-backed ranks are already fixed, while too few no-email accounts remain to create the missing overlap. Addressing that case requires either better normalized email agreement, delaying commitment, or allowing unmatched Reference-ID ranks—not only proprietary fallbacks—to participate in the adaptive allocation. The last option increases late-anchor conflict risk and needs separate validation.
