# AI-Powered Loan Decision & Explanation System

An AI-Assisted Underwriting & Explanation platform for a mid-sized Indian retail bank, built for MBA-level Foundations of AI for Business.

---

## Architectural Principle: Strict Decoupling

```
                                 [ Loan Application Intake ]
                                              │
                                              ▼
                             [ Stage 0: Input Gate & Guardrails ]
                             - Schema bounds (Tenure 2-20, CIBIL 300-900)
                             - Prompt-injection neutralizer
                             - Negative collateral detector (DATA_HOLD)
                             - Out-of-Distribution (OOD_RANGE) detector
                                              │
                                              ▼
                        [ Stage 1: Deterministic Ratio Engine ]
                        - Pure Python math: LTI, Asset Coverage, Total Assets
                        - Policy breach detection (CP-2.1, CP-3.1, CP-4.1)
                        - Exception pattern routing (CP-4.2, CP-2.2)
                                              │
                                              ▼
                         [ Stage 2: Predictive Risk Scoring ]
                         - Scikit-Learn RandomForestClassifier (100 trees)
                         - Generates P(Approve) on 11 raw features
                         - Local SHAP TreeExplainer attributions
                                              │
                                              ▼
                       [ Stage 3: Decision Routing Hierarchy ]
                       - Strictly deterministic rule order:
                         1. Schema / bounds failure   ──► BLOCKED
                         2. Negative asset valuation  ──► DATA_HOLD
                         3. Any exception or flag     ──► ESCALATE
                         4. P > 0.85 & 0 breaches     ──► AUTO_APPROVE
                         5. P < 0.20 & >= 1 breach    ──► AUTO_REJECT
                         6. All remaining cases       ──► ESCALATE
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
       [ Auto-Tracks (Approve / Reject) ]                 [ Human Underwriter Desk ]
                    │                                     - Mandatory rationale (>= 20 chars)
                    │                                     - CP-7.2 condition menu (C1-C4)
                    │                                     - Four-Eyes approval for > ₹2 Cr
                    │                                     - Data correction re-run
                    └─────────────────────────┬─────────────────────────┘
                                              │
                                              ▼
                             [ Stage 5: Grounded RAG Engine ]
                             - Hybrid policy retrieval (Breach-mapped + TF-IDF)
                             - Free-tier Groq LLM (llama-3.3-70b-versatile)
                             - Strict Python post-validation:
                               * Whitelist clause verification
                               * Verbatim substring citation check (<= 40 words)
                               * Numeric token containment check
                               * Forbidden content regex (no fees/rates/demographics)
                             - Side-by-side highlighted RAG citation viewer
                                              │
                                              ▼
                        [ Tamper-Evident SHA-256 Audit Chain ]
                        - Immutable event log in SQLite
```

---

## Quick Start & Installation

### 1. Requirements & Dependencies
Ensure Python 3.10+ is installed. Install the requirements:
```bash
pip install -r requirements.txt
```

### 2. Required Input Files
The system requires the following four files in the project root:
- `loan_approval_dataset.csv`: Historical 4,269-row underwriting dataset.
- `credit-policy.md`: Runtime credit policy and thresholds.
- `letter-templates.md`: Customer letter and credit memo skeletons.
- `credentials.json`: Groq API key (`groq_api_key`) and optional Gmail credentials.

### 3. Run the Application
Launch the Streamlit portal from `credit-risk-app`:
```bash
streamlit run app.py
```

### 4. Run Automated Tests
```bash
pytest tests/ -v
```

---

## Acceptance Criteria Verification

- [x] **Decoupled Scoring**: Scoring and routing are 100% deterministic Python and Scikit-Learn Random Forest. Zero LLM in quantitative decisions.
- [x] **Strict RAG Post-Validation**: Every LLM narrative passes substring citation check, numeric whitelist check, and fair-lending prohibitions.
- [x] **Collateral Shortfall Handling**: Prime applicants with high CIBIL and asset shortfall are explained as collateral coverage matters (CP-4.1/CP-4.2), never as credit score failures.
- [x] **Negative Asset Gate**: Negative collateral never reaches ML or LLM; routes to `DATA_HOLD`.
- [x] **Four-Eyes Governance**: High-ticket loans (> ₹2 Cr) or counter-model decisions require two independent reviewers.
- [x] **Tamper-Evident Audit Chain**: Append-only SHA-256 hash-chained log detects any unauthorized database record edits.
- [x] **Appendix C Hardening**: All 15 peer-challenge stress tests pass cleanly.

---

## Known Limitations

1. **Historical Approvals Proxy**: Historical data reflects underwriting approval labels rather than true post-disbursal defaults.
2. **Probability Calibration**: Raw Random Forest probabilities skew towards 0 and 1; optional isotonic/sigmoid calibration toggle is provided.
3. **Macroeconomic Shocks**: Model does not ingest real-time macroeconomic interest rate shocks or sector-specific credit contractions.
