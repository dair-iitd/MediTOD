var residence_config = [
    {
        "name": "value",
        "type": "datalist",
        "values": [
            {
                "code": "big city",
                "description": "",
                "display_name": "Big City"
            },
            {
                "code": "mininsuburbsg sector",
                "description": "",
                "display_name": "Suburbs"
            },
            {
                "code": "rural",
                "description": "",
                "display_name": "Rural"
            }
        ],
        "label": "Add details for patient's residence like urban/rural and apartment/house",
        "add_checkbox": ["inquire"],
    },
    {
        "name": "status",
        "type": "select",
        "values": ["Yes", "No"],
        "label": "Status",
    },
    {
        "name": "household size",
        "type": "text",
        "label": "Size of the patient's household",
    },
    {
        "name": "other",
        "type": "text",
        "label": "Any additional information missing from the above fields",
    }
]
