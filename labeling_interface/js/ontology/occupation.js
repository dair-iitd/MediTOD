var occupation_config = [
    {
        "name": "value",
        "type": "datalist",
        "values": [
            {
                "code": "construction",
                "description": "",
                "display_name": "Construction"
            },
            {
                "code": "mining sector",
                "description": "",
                "display_name": "Mining sector"
            },
            {
                "code": "daycare",
                "description": "",
                "display_name": "Daycare"
            },
            {
                "code": "agriculture",
                "description": "",
                "display_name": "Agriculture"
            }
        ],
        "label": "Add patient's occupation details like job sector or job title"
    },
    {
        "name": "status",
        "type": "select",
        "values": ["Yes", "No"],
        "add_checkbox": ["inquire"],
        "label": "Has the patient works/worked at the above occupation?",
    },
    {
        "name": "exposure",
        "type": "text",
        "add_checkbox": ["inquire"],
        "label": "Are there any hazards / substances / dangers to which the patient got exposed at work?",
    },
    {
        "name": "other",
        "type": "text",
        "label": "Any additional information missing from the above fields",
    }
]
