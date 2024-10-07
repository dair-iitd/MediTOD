var tracking_slots = [
    'symptom', 'symptom (dermatology)',
    'medical history', 'family history', 'exposure',
    'disease', "medication", "habit", "travel"
];

var good_values = [
    "Yes",
    "The patient still suffers from the condition.",
    "The patient sufferred from the condition in the past.",
    "currently taking",
    "took in the past",
    // legacy
    "Patient still suffers from the condition",
    "Patient sufferred from the condition in the past",
];

var bad_values = [
    "No",
    "The patient did not suffer from the condition in the past.",
    "no",
    // legacy
    "Patient did not suffer from the condition in the past"
];

var not_sure_values = [
    "Maybe",
    "Not sure",
    "The patient is not sure about the status of the condition.",
    // legacy
    "Patient is not sure about the status of the condition"
];