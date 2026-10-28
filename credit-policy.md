# Retail Credit Policy: Home & Personal Loan Underwriting (Starter Version 1.0)

Currency: INR. Applies to retail loan applications screened by the AI-Assisted Underwriting System. Thresholds in the parameters block below are read by the system at runtime.

## Policy Parameters

```json
{
  "cibil_min": 300,
  "cibil_max": 900,
  "cibil_standard_floor": 550,
  "tenure_min_years": 2,
  "tenure_max_years": 20,
  "lti_max": 4.0,
  "asset_coverage_min": 1.0,
  "subprime_short_tenure_max_years": 4,
  "subprime_min_coverage": 1.0,
  "collateral_shortfall_prime_cibil_min": 580,
  "auto_approve_prob_min": 0.85,
  "auto_reject_prob_max": 0.20,
  "four_eyes_loan_threshold_inr": 20000000,
  "min_loan_amount_inr": 100000,
  "min_income_inr": 100000,
  "loan_amount_hard_cap_inr": 100000000
}
```

## 1. Data Integrity and Input Bounds

### [CP-1.1] Asset Valuation Integrity
All asset values must be zero or positive. A negative asset value indicates an encumbered or underwater asset or a data-entry error. Such applications must not be scored automatically and must be verified by a Senior Credit Underwriter before any decision.

### [CP-1.2] Input Bounds
Loan tenure must be between 2 and 20 years and the credit score must be between 300 and 900. Applications outside these bounds cannot be processed. Values outside the range of historical experience must be referred to a Senior Credit Underwriter.

## 2. Credit Score

### [CP-2.1] Credit Score Floor
The standard minimum CIBIL score is 550. Applications with a CIBIL score below 550 do not qualify for automated approval.

### [CP-2.2] Short-Tenure Subprime Exception
An application with a CIBIL score below 550 may be considered only where the loan tenure is between 2 and 4 years and Total Asset Coverage is greater than 1.0x. Every such application requires Senior Credit Underwriter sign-off.

## 3. Leverage

### [CP-3.1] Loan-to-Income Limit
The Loan-to-Income ratio, calculated as the loan amount divided by annual income, must not exceed 4.0x.

## 4. Collateral

### [CP-4.1] Minimum Total Asset Coverage
Total Asset Coverage, calculated as total assets divided by the loan amount, must be at least 1.0x. Total assets are the sum of residential, commercial, luxury and bank liquid assets.

### [CP-4.2] Collateral Shortfall for Prime Applicants
An applicant with a CIBIL score of 580 or above whose only breach is Total Asset Coverage below 1.0x must be referred to a Senior Credit Underwriter, who may approve the application with conditions to cure the shortfall.

## 5. Decision Routing

### [CP-5.1] Automated Approval
An application may be approved automatically only where the model approval probability is above 85%, there are no policy breaches, and there are no data or exception flags.

### [CP-5.2] Automated Rejection
An application may be rejected automatically only where the model approval probability is below 20% and at least one clear breach of clause CP-1.2, CP-2.1, CP-3.1 or CP-4.1 exists. A low model score alone is not sufficient reason to reject.

### [CP-5.3] Referral to Underwriter
All applications that are neither automatically approved nor automatically rejected, and all applications with an exception pattern or data flag, must be referred to a Senior Credit Underwriter.

## 6. Human Checkpoint and Governance

### [CP-6.1] Mandatory Sign-Off
No referred application may be approved, rejected or disbursed without the recorded decision of a Senior Credit Underwriter, including a written rationale of at least 20 characters.

### [CP-6.2] Second Approval
A second, different Senior Credit Underwriter must confirm any approval of a referred application above INR 2 Crore, and any decision that contradicts the model direction.

## 7. Adverse Action and Conditions

### [CP-7.1] Adverse Action Notice
An adverse action notice must state the principal reasons for the decision. Each reason must be tied to a specific policy clause with the measured value and the threshold. A reason that has not been verified by the ratio engine must never be stated.

### [CP-7.2] Approval Conditions
Approvals with conditions may be drawn only from this list: C1, reduce the loan amount to meet the leverage and coverage limits; C2, shorten the loan tenure within 2 to 20 years; C3, provide additional collateral so that Total Asset Coverage reaches at least 1.0x; C4, complete document verification.

## 8. Fair Lending and Communication

### [CP-8.1] Fair Lending
Decisions and customer explanations must rest on financial factors only. Education, number of dependents and employment type must not be cited as reasons in any customer communication.

### [CP-9.1] No Pricing in Automated Communication
Interest rates, fees and instalment amounts are not determined by this system. Automated communication must not state or imply any interest rate, fee, instalment amount or guarantee of approval.

### [CP-9.2] Plain Language
Customer communication must be written in plain, respectful language.

## 9. Records

### [CP-10.1] Record Retention
Every decision must be recorded with its timestamp, model version, policy version and the identity of any human reviewer.
