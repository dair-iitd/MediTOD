
var habit_config = [
    {
        "name": "value",
        "type": "datalist",
        "values": [
            {
                "code": "alcoholism",
                "description": "The patient drinks alcohol in excess.",
                "display_name": "alcoholism"
            },
            {
                "code": "former smoker",
                "description": "The patient is a former smoker.",
                "display_name": "former smoker"
            },
            {
                "code": "smoking",
                "description": "The patient smokes cigarettes.",
                "display_name": "smoking"
            },
            {
                "code": "secondhand cigarette",
                "description": "The patient does not smoke but is exposed to second hand cigarettes smokes on daily basis.",
                "display_name": "secondhand cigarette"
            },
            {
                "code": "regular energy drinks",
                "description": "The patient consumes energy drinks regularly.",
                "display_name": "regular energy drinks"
            },
            {
                "code": "regular exercise",
                "description": "The patient exercises regularly (4 times per week or more).",
                "display_name": "regular exercise"
            },
            {
                "code": "regular coffee tea",
                "description": "The patient drinks coffee or tea regularly.",
                "display_name": "regular coffee tea"
            },
            {
                "code": "marijuana",
                "description": "The patient smokes marijuana / cannabis .",
                "display_name": "marijuana"
            },
            {
                "code": "recreational drugs",
                "description": "The patient takes recreational drugs.",
                "display_name": "recreational drugs"
            },
        ],
        "label": "Select activity/activities"
    },
    {
        "name": "status",
        "type": "select",
        "values": ["Yes", "No", "Maybe"],
        "label": "Has/had patient formed a habit/addiction to the selected activities?",
        "add_text": ["inform"],
        "aux_value": {
            "tag": "criterion",
            "placeholder": "criterion"
        },
        "default": { "doctor": "disabled" },
    },
    {
        "name": "frquency",
        "type": "text",
        "label": "How frequently does the patient engages into the selected activity?",
        "add_checkbox": ["inquire"],
    },
    {
        "name": "when",
        "type": "text",
        "label": "When did the patient pick up the activities?",
        "add_checkbox": ["inquire"],
    },
    {
        "name": "other",
        "type": "text",
        "label": "Enter any additional information missing from above fields.",
    }
]
