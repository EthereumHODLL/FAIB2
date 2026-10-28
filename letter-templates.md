# Letter and Memo Templates

## Customer Approval Letter

Subject: Your Loan Application {{APPLICATION_ID}}: {{HEADLINE}}

Dear {{APPLICANT_NAME}},

Date: {{DATE}}

{{NARRATIVE}}

Loan amount applied for: {{LOAN_AMOUNT}}
Tenure applied for: {{TENURE}} years

{{CONDITIONS_BLOCK}}

{{NEXT_STEPS_BLOCK}}

This letter is not a disbursal commitment until all conditions are met and documents verified.

Regards,
Retail Credit Team

## Customer Adverse Action Notice

Subject: Update on Your Loan Application {{APPLICATION_ID}}

Dear {{APPLICANT_NAME}},

Date: {{DATE}}

{{NARRATIVE}}

Principal reasons for this decision:
{{REASONS_BLOCK}}

{{NEXT_STEPS_BLOCK}}

You may contact the bank to discuss this decision or to submit additional information.

Regards,
Retail Credit Team

## Internal Credit Risk Memo

Case: {{APPLICATION_ID}} | Date: {{DATE}} | Track: {{TRACK}} | Model: {{MODEL_VERSION}} | Policy: {{POLICY_VERSION}}

1. Summary: {{SECTION_SUMMARY}}
2. Applicant metrics: {{METRICS_TABLE}}
3. Risk drivers (model attribution): {{SECTION_RISK_DRIVERS}}
4. Policy findings (with citations): {{REASONS_BLOCK}}
5. Model signal: {{SECTION_MODEL_SIGNAL}}
6. Flags and exceptions: {{FLAGS_BLOCK}}
7. Human reviewer decision: {{HUMAN_DECISION_BLOCK}}
8. Recommendation: {{SECTION_RECOMMENDATION}}
