var exposure_config = [
    {
        "name": "value",
        "type": "datalist",
        "values": [
            {
                "code": "ebola",
                "description": "The patient was in ebola infested area.",
                "display_name": "Ebola"
            },
            {
                "code": "pertussis",
                "description": "The patient was in whooping cough/pertussis infected area.",
                "display_name": "pertussis"
            },
            {
                "code": "person",
                "description": "The patient was in contact with someone with similar symptom.",
                "display_name": "person"
            },
            {
                "code": "allergy",
                "description": "The patient was in contact with something he/she is allergic to.",
                "display_name": "allergy"
            },
            {
                "code": "pets",
                "description": "The patient has pet(s).",
                "display_name": "pets"
            },
            {
                "code": "dust",
                "description": "The patient has exposure to dust.",
                "display_name": "dust"
            },
            {
                "code": "chemicals",
                "description": "The patient has exposure to chemicals.",
                "display_name": "chemicals"
            },
        ],
        "label": "Select exposure factor(s)"
    },
    {
        "name": "status",
        "type": "select",
        "values": ["Yes", "No", "Maybe"],
        "label": "Was the patient exposed to the selected exposure factor?"
    },
    {
        "name": "when",
        "type": "text",
        "add_checkbox": ["inquire"],
        "label": "When was the patient exposed to the selected factor?"
    },
    {
        "name": "where",
        "type": "text",
        "add_checkbox": ["inquire"],
        "label": "Where was the patient exposed to the selected factor e.g. at work or at home?"
    },
    {
        "name": "other",
        "type": "text",
        "label": "Any additional information missing from the selected fields"
    }
]
