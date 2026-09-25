"""Validate zero-shot document-type classification across many doc genres."""
from laya import Router

router = Router()

DOC_QUESTION = {
    "document_type": {
        "type": "choice",
        "instructions": "What type of document is this? Classify by its structure, purpose and genre.",
        "criteria": {
            "resume": "CV, resume: a person's work experience, education, skills, contact details for job applications",
            "research_paper": "academic/scientific paper: abstract, methodology, citations, references, experiments",
            "news_article": "journalism: reporting of recent events, headline, byline, quotes from sources",
            "invoice": "bill or invoice: amounts due, line items, payment terms, invoice number, taxes",
            "manual": "instruction/technical manual or documentation: how to install, operate, configure, troubleshoot",
            "book": "book or book chapter: narrative prose, fiction or non-fiction long-form chapters",
            "recipe": "recipe or cookbook page: ingredients list and cooking steps",
            "magazine": "magazine article: lifestyle, entertainment, interviews, feature stories, ads",
            "legal_contract": "legal document: agreement, contract, terms and conditions, clauses, parties",
            "business_report": "business/technical report: executive summary, KPIs, quarterly results, analysis",
            "personal_letter": "personal letter, email or message correspondence",
            "other": "anything that does not fit the above categories",
        },
    }
}

samples = {
    "resume": "John Doe\nSoftware Engineer\njohn.doe@email.com | +1-555-0123\n\nEXPERIENCE\nSenior Developer, Acme Corp (2019-Present)\n- Led team of 5 engineers building microservices\nEDUCATION\nB.S. Computer Science, MIT, 2015\nSKILLS: Python, Kubernetes, AWS",
    "research_paper": "Abstract: We present a novel transformer architecture that reduces computational complexity from O(n^2) to O(n log n). Our experiments on GLUE benchmarks demonstrate... 1. Introduction Attention mechanisms (Vaswani et al., 2017) have become... References [1] Devlin, J. et al. BERT. NAACL 2019.",
    "news_article": "BREAKING: City Council Approves New Transit Plan. By Jane Smith, Staff Writer. Published Sep 25, 2026. The city council voted 7-2 on Tuesday to approve the $2.4 billion light rail project, ending years of debate. 'This is a historic moment,' said Mayor...",
    "invoice": "INVOICE #4411\nBill To: Acme LLC\nDate: 2026-09-01\nDue: Net 30\n\nDescription | Qty | Amount\nConsulting services | 10h | $1,500.00\nHosting (Sept) | 1 | $99.00\nSubtotal: $1,599.00\nTax (8%): $127.92\nTOTAL DUE: $1,726.92",
    "manual": "Quick Start Guide — Model XR-200 Router\n1. Unbox the device and connect the power adapter.\n2. Press and hold the reset button for 10 seconds to restore factory settings.\n3. Open a browser and navigate to 192.168.0.1\nTroubleshooting: If the LED blinks red, check the WAN cable...",
    "book": "Chapter 7: The Long Winter. The snow had fallen for three days without pause, and Elena watched from the tower window as the valley disappeared beneath a white silence. She thought of her father's warning, spoken years ago...",
    "recipe": "Grandma's Apple Pie\nIngredients:\n- 6 cups sliced Granny Smith apples\n- 3/4 cup sugar\n- 2 tbsp all-purpose flour\n- 1 tsp cinnamon\n- 2 pie crusts\nInstructions: Preheat oven to 425F. Mix apples, sugar, flour and cinnamon. Fill crust, cover with top crust, bake 40-45 min until golden.",
    "magazine": "STYLE & CULTURE — 10 Must-Have Looks This Fall. From oversized blazers to vintage denim, our editors round up the season's hottest trends. Plus: an exclusive interview with pop sensation Luna on her new album, beauty secrets and life on tour.",
    "legal_contract": "SERVICES AGREEMENT. This Agreement is entered into as of Jan 1, 2026 by and between Party A ('Provider') and Party B ('Client'). 1. SCOPE OF SERVICES. Provider shall deliver the services described in Exhibit A. 2. TERMINATION. Either party may terminate with 30 days written notice. IN WITNESS WHEREOF...",
    "business_report": "Q3 2026 Earnings Report — Executive Summary. Revenue grew 14% YoY to $48.2M, driven by strong enterprise segment performance. Operating margin expanded 230bps. Key risks include supply chain constraints. Outlook: we raise full-year guidance to...",
}

correct = 0
for expected, text in samples.items():
    r = router.predict(text, DOC_QUESTION)
    got = r["answers"]["document_type"]["choice"]
    conf = r["answers"]["document_type"].get("answer_confidence")
    ok = got == expected
    correct += ok
    print(f"{'OK ' if ok else 'MISS'} expected={expected:16s} got={got:16s} conf={conf:.3f} model={r['routing']['model']}")

print(f"\n{correct}/{len(samples)} correct")
