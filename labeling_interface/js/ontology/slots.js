var slot_list = [
    {
        "value": "basic information",
        "help_message": "The utterance contains the details (slot-values) for basic information about the patient - name, age, and sex.",
        "parent_intents": ['inform', 'inquire', 'other'],
        "config": basic_config,
    },
    {
        "value": "symptom",
        "help_message": "The utterance contains the details (slot-values) for a symptom. A symptom carries a value (like cough or fever) and additional fields (start of the symptom, its nature, and so on).",
        "parent_intents": ['inform', 'inquire', 'other'],
        "config": symptom_config,
    },
    {
        "value": "symptom (dermatology)",
        "help_message": "The utterance contains the details (slot-values) for a dermatological symptom, i.e., symptom related to skin, nails, and hair. A symptom carries a value (like a rash) and additional fields (color, size, swelling).",
        "parent_intents": ['inform', 'inquire', 'other'],
        "config": symptom_dermatology_config,
    },
    {
        "value": "disease",
        "help_message": "In the utterance, the doctor is diagnosing a disease.",
        "parent_intents": ['inform', 'inquire', 'other', 'diagnosis'],
        "config": disease_config,
    },
    {
        "value": "exposure",
        "help_message": "The utterance contains the details (slot-values) for the situation where the patient is likely to be exposed to a harmful situation. This includes contact with allergic substances, dust, chemicals, and infected persons.",
        "parent_intents": ['inform', 'inquire', 'other'],
        "config": exposure_config,
    },
    {
        "value": "habit",
        "help_message": "The utterance contains the details (slot-values) for a habit/addiction related to the patient. A habit is an activity that the patient performs regularly. This ranges from daily exercise, tea, and coffee to smoking, alcoholism, and marijuana abuse.",
        "parent_intents": ['inform', 'inquire', 'other'],
        "config": habit_config,
    },
    {
        "value": "medical history",
        "help_message": "The utterance contains the details (slot-values) of the patient's medical history. It can describe a symptom/disease/surgery/allergy the patient previously had. Note that the disease information in the medical history differs from that in the Disease slot. Here it describes a past medical condition rather than a doctor's diagnosis.",
        "parent_intents": ['inform', 'inquire', 'other'],
        "config": medical_history_config,
    },
    {
        "value": "medication",
        "help_message": "The utterance contains the details (slot-values) for medication, specific like Tylenol or general like anti-psychotic drugs. The utterance can also discuss additional information, like what the medicine is for and when the patient is on the medication. The doctor can also inform drugs to the patient.",
        "parent_intents": ['inform', 'inquire', 'other'],
        "config": medication_config,
    },
    {
        "value": "medical test",
        "help_message": "The utterance contains the details (slot-values) for a medical test like ECG or CAT scan. The doctor can inquire about a medical examination that the patient has had or inform the patient to have a test done.",
        "parent_intents": ['inform', 'inquire', 'other'],
        "config": medical_test_config,
    },
    {
        "value": "family history",
        "help_message": "The utterance contains the details (slot-values) of medical conditions prevalent through the patient's family. This includes diseases like asthma, heart issues, cancer, etc.",
        "parent_intents": ['inform', 'inquire', 'other'],
        "config": family_history_config,
    },
    {
        "value": "occupation",
        "help_message": "The utterance contains the details (slot-values) for the patient's occupation, like the teacher, trucker, factory worker, etc.",
        "parent_intents": ['inform', 'inquire', 'other'],
        "config": occupation_config,
    },
    {
        "value": "residence",
        "help_message": "The utterance contains the details (slot-values) for the patient's residence, like urban, rural, suburban, etc.",
        "parent_intents": ['inform', 'inquire', 'other'],
        "config": residence_config
    },
    {
        "value": "travel",
        "help_message": "The utterance contains the details (slot-values) for any travel undertaken by the patient. It may include details like the time of the trip, its location, and frequency.",
        "parent_intents": ['inform', 'inquire', 'other'],
        "config": travel_config,
    },
    {
        "value": "medical discussion",
        "help_message": "The utterance is from a chit-chat about a medical topic (e.g., pulmonary embolism).",
        "parent_intents": ['chit-chat'],
        "config": medical_discussion_config,
    },
    {
        "value": "non-medical discussion",
        "help_message": "The utterance is from a chit-chat about a non-medical topic (e.g., living conditions).",
        "parent_intents": ['chit-chat'],
        "config": non_medical_discussion_config,
    },
    {
        "value": "other",
        "help_message": "Any other slot the utterance may contain the details (slot-values) apart from the above.",
        "parent_intents": ['inform', 'inquire', 'other'],
        "config": other_config,
    },
    {
        "value": "diseases/conditions",
        "help_message": "Differential diagnosis for the patient",
        "parent_intents": ['differential'],
        "config": ddx_config,
    }
]


var slot_config = {
    "question": "What is the slot type? ",
    "help_message": "Identify slot types relevant to the intent. For example, in the utterance 'Are you feeling any pain?' the speaker intent is 'inquire' and the slot type is the 'symptom'. After selecting the appropriate slot type, values ('pain') can be filled in.",
    "slot_list": slot_list,
}
