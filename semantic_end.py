semantic_end = [
        {
            "id": "sem_end_01",
            "description": "pilot : plane :: sailor : ? (end)",
            "bert_prompt": "A pilot works for an airline, just like a sailor works for a [MASK].",
            "distilbert_prompt": "A pilot works for an airline, just like a sailor works for a [MASK].",
            "roberta_prompt": "A pilot works for an airline, just like a sailor works for a <mask>.",
            "t5_input": "fill mask: A pilot works for an airline, just like a sailor works for a <extra_id_0>.",
            "gpt_prompt": "A pilot works for an airline. Similarly, a sailor works for a",
            "correct_answers": ["ship", "boat", "vessel"]
        },
        {
            "id": "sem_end_02",
            "description": "doctor : hospital :: teacher : ? (end)",
            "bert_prompt": "A doctor works in a hospital while a teacher works in a [MASK].",
            "distilbert_prompt": "A doctor works in a hospital while a teacher works in a [MASK].",
            "roberta_prompt": "A doctor works in a hospital while a teacher works in a <mask>.",
            "t5_input": "fill mask: A doctor works in a hospital while a teacher works in a <extra_id_0>.",
            "gpt_prompt": "A doctor works in a hospital, while a teacher works in a",
            "correct_answers": ["school", "classroom"]
        },
        {
            "id": "sem_end_03",
            "description": "chef : kitchen :: teacher : ? (end)",
            "bert_prompt": "A chef works in a kitchen just like a teacher works in a [MASK].",
            "distilbert_prompt": "A chef works in a kitchen just like a teacher works in a [MASK].",
            "roberta_prompt": "A chef works in a kitchen just like a teacher works in a <mask>.",
            "t5_input": "fill mask: A chef works in a kitchen just like a teacher works in a <extra_id_0>.",
            "gpt_prompt": "A chef works in a kitchen, similarly a teacher works in a",
            "correct_answers": ["classroom", "school"]
        }
]