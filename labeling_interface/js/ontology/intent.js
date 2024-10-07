var intent_list = [
    {
        "value": "inform",
        "help_message": "The speaker provides specific information (like symptoms, medical history, etc.). This can be in response to an inquiry or can be impromptu. Select if 'specific information' is necessary to make an educated diagnosis.",
        "applicable_speakers": ["doctor", "patient"]
    },
    {
        "value": "inquire",
        "help_message": "The speaker wants to gather specific information (like symptoms, medical history, etc.). Select if 'specific information' is necessary to make an educated diagnosis.",
        "applicable_speakers": ["doctor", "patient"]
    },
    {
        "value": "diagnosis",
        "help_message": "The doctor makes a diagnosis of a disease.",
        "applicable_speakers": ["doctor"] },
    {
        "value": "salutations",
        "help_message": "The speaker wants to convey a hello/goodbye message.",
        "applicable_speakers": ["doctor", "patient"]
    },
    {
        "value": "chit-chat",
        "help_message": "The speaker makes a casual conversation. Here, information in the utterance is unlikely to help an educated diagnosis.",
        "applicable_speakers": ["doctor", "patient"]
    },
    {
        "value": "nod_prompt",
        "help_message": "The speaker does not provide information relevant to medical diagnosis. Instead, the speaker conveys attention, understanding, or agreement with phrases like  - Okay, Yeah, uh-huh, etc.",
        "applicable_speakers": ["doctor", "patient"]
    },
    {
        "value": "other",
        "help_message": "Any other intent the speaker may have apart from the above.",
        "applicable_speakers": ["doctor", "patient"]
    },
    {
        "value": "differential",
        "help_message": "Add differential diagnosis based on the dialog",
        "applicable_speakers": ["control"]
    },
]


var intent_config = {
    "question": "What is the intent of the speaker? ",
    "help_message": "Identify the speaker's intention behind an utterance. For example, in the utterance 'Are you feeling any pain?' the speaker's intent is 'inquire'.",
    "intent_list": intent_list,
}