var all_travel_loc = [
    {
        "code": "Asia",
        "display_name": "",
        "description": ""
    },
    {
        "code": "Caraibes",
        "display_name": "",
        "description": ""
    },
    {
        "code": "Central America",
        "display_name": "",
        "description": ""
    },
    {
        "code": "Europe",
        "display_name": "",
        "description": ""
    },
    {
        "code": "No",
        "display_name": "",
        "description": ""
    },
    {
        "code": "North Africa",
        "display_name": "",
        "description": ""
    },
    {
        "code": "North America",
        "display_name": "",
        "description": ""
    },
    {
        "code": "Oceania",
        "display_name": "",
        "description": ""
    },
    {
        "code": "South Africa",
        "display_name": "",
        "description": ""
    },
    {
        "code": "South America",
        "display_name": "",
        "description": ""
    },
    {
        "code": "South East Asia",
        "display_name": "",
        "description": ""
    },
    {
        "code": "West Africa",
        "display_name": "",
        "description": ""
    }
]

var travel_config = [
    {
        "name": "status",
        "type": "select",
        "values": ["Yes", "No"],
        "add_checkbox": ["inquire"],
        "label": "Has the patient travelled recently?",
    },
    {
        "name": "destination",
        "type": "datalist",
        "values": all_travel_loc,
        "add_checkbox": ["inquire"],
        "label": "Where has the patient travelled to?",
        // "default": "none"
    },
    {
        "name": "when",
        "type": "text",
        "add_checkbox": ["inquire"],
        "label": "When did the patient travel?",
    },
    {
        "name": "frequency",
        "type": "text",
        "add_checkbox": ["inquire"],
        "label": "How frequently does the patient travel?",
    },
    {
        "name": "other",
        "type": "text",
        "label": "Enter additional information about travel",
    },
]
