"""Verify Laya works end-to-end: routing + real prediction (README quickstart)."""
from laya import Router

router = Router()  # lazy: downloads only the checkpoint that gets routed to

state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
questions = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}

result = router.predict(state, questions)
print("department :", result["answers"]["department"]["choice"])
print("urgency    :", result["answers"]["urgency"])
print("churn_risk :", result["answers"]["churn_risk"])
print("routing    :", result["routing"])
