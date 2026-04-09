semantic_middle = [

           {
            "id": "sem_mid_01",
            "description": "role-to-workplace (middle)",
            "bert_prompt": "A pilot works for a [MASK], just like a sailor works for a ship.",
            "distilbert_prompt": "A pilot works for a [MASK], just like a sailor works for a ship.",
            "roberta_prompt": "A pilot works for a <mask>, just like a sailor works for a ship.",
            "t5_input": "fill mask: A pilot works for a <extra_id_0>, just like a sailor works for a ship.",
            "gpt_prompt": "A pilot works for a , just like a sailor works for a ship. Fill in the blank:",
            "correct_answers": ["airline", "airline company"]
        },
        {
            "id": "sem_mid_02",
            "description": "object property (middle)",
            "bert_prompt": "The sun is a [MASK] star in our galaxy.",
            "distilbert_prompt": "The sun is a [MASK] star in our galaxy.",
            "roberta_prompt": "The sun is a <mask> star in our galaxy.",
            "t5_input": "fill mask: The sun is a <extra_id_0> star in our galaxy.",
            "gpt_prompt": "Complete: The sun is a ____ star in our galaxy.",
            "correct_answers": ["medium-sized", "typical"]
        },
        {
            "id": "sem_mid_03",
            "description": "role-to-tool (middle)",
            "bert_prompt": "A surgeon holds a [MASK] while performing surgery.",
            "distilbert_prompt": "A surgeon holds a [MASK] while performing surgery.",
            "roberta_prompt": "A surgeon holds a <mask> while performing surgery.",
            "t5_input": "fill mask: A surgeon holds a <extra_id_0> while performing surgery.",
            "gpt_prompt": "A surgeon holds a ____ while performing surgery. Fill in the blank:",
            "correct_answers": ["scalpel", "forceps"]
        }
]