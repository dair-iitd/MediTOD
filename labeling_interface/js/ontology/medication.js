var medication_config = [
    {
        "name": "value",
        "type": "datalist",
        "values": [
            {
                "code": "antipsychotic medication",
                "description": "",
                "display_name": "Antipsychotic medication"
            },
            {
                "code": "corticosteroids",
                "description": "",
                "display_name": "Corticosteroids"
            },
            {
                "code": "noacs",
                "description": "",
                "display_name": "Noacs"
            },
            {
                "code": "calcium channel breaker",
                "description": "",
                "display_name": "Calcium channel breaker"
            },
            {
                "code": "hormone intake",
                "description": "",
                "display_name": "Hormone intake"
            },
            {
                "code": "intravenous drugs",
                "description": "",
                "display_name": "Intravenous drugs"
            },
            {
                "code": "stimulant drugs",
                "description": "",
                "display_name": "Stimulant drugs"
            },
            {
                "code": "nsaids",
                "description": "",
                "display_name": "Nsaids"
            },
            {
                "code": "vasodilators",
                "description": "",
                "display_name": "Vasodilators"
            }
        ],
        "add_checkbox": ["enquire"],
        "label": "Enter medication"
    },
    {
        "name": "status",
        "type": "select",
        "values": ["currently taking", "took in the past", "no"],
        "label": "Medication Status"
    },
    {
        "name": "response to",
        "type": "text",
        "add_checkbox": ["enquire"],
        "label": "For which condition/symptom is medication for?",
        "add_checkbox": ["inquire"],
    },
    {
        "name": "since when",
        "type": "text",
        "add_checkbox": ["enquire"],
        "label": "Since when did the patient start taking the medication?",
        "add_checkbox": ["inquire"],
    },
    {
        "name": "frequency",
        "type": "text",
        "add_checkbox": ["enquire"],
        "label": "How frequently does the patient take the medication?",
        "add_checkbox": ["inquire"],
    },
    {
        "name": "impact",
        "type": "select",
        "add_checkbox": ["enquire"],
        "values": ["Yes", "No", "Maybe"],
        "label": "Did the medication help the patient?",
        "add_checkbox": ["inquire"],
    },
    {
        "name": "other",
        "type": "text",
        "label": "Any additional information missing from above field."
    },
]


var medical_test_config = [
    {
        "name": "name",
        "type": "text",
        "label": "Enter medical test"
    },
    {
        "name": "status",
        "type": "select",
        "values": ["Yes", "No"],
        "label": "Does patient has the results for the medical test?",
    },
    {
        "name": "when",
        "type": "text",
        "label": "When did the patient had the medical test done?",
    },
    {
        "name": "other",
        "type": "text",
        "label": "Any additional information missing from above field"
    },
]
