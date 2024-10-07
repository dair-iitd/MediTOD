var symptom_dermatology_config = [
    {
        "name": "status",
        "type": "select",
        "values": ["Yes", "No", "Not sure"],
        "label": "Does the patient have any lesions, redness or problems on the skin?",
        "default": { "patient": "Yes" },
    },
    {
        "name": "swollen",
        "type": "select",
        "values": ["Yes", "No"],
        "label": "Is the rash swollen?",
        "chkbox_on": ["inquire"],
    },
    {
        "name": "size",
        "type": "select",
        "values": ["Yes", "No"],
        "label": "Is the lesion (or are the lesions) larger than 1cm?",
        "chkbox_on": ["inquire"],
    },
    {
        "name": "peel off",
        "type": "select",
        "values": ["Yes", "No"],
        "label": "Do the lesions peel off?",
        "chkbox_on": ["inquire"],
    },
    {
        "name": "itching",
        "type": "select",
        "values": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        "label": "How severe is the itching on scale of 10?",
        "chkbox_on": ["inquire"],
    },
    {
        "name": "location",
        "type": "datalist",
        "values": all_symptom_locations,
        "label": "Where is the affected region located?",
        "add_checkbox": ["inquire"],
    },
    {
        "name": "color",
        "type": "datalist",
        "values": [
            {
                "code": "dark",
                "description": "",
                "display_name": "Dark"
            },
            {
                "code": "yellow",
                "description": "",
                "display_name": "Yellow"
            },
            {
                "code": "pale",
                "description": "",
                "display_name": "Pale"
            },
            {
                "code": "pink",
                "description": "",
                "display_name": "Pink"
            },
            {
                "code": "red",
                "description": "",
                "display_name": "Red"
            },
        ],
        "label": "What color is the rash?",
        "add_checkbox": ["inquire"],
    },
    {
        "name": "other",
        "type": "text",
        "label": "Any additional information missing from the above fields"
    }
]
