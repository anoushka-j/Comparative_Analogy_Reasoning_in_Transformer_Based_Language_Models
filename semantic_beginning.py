semantic_beginning = [

{
            "id": "sem_beg_01",
            "description": "role-to-worker (beginning)",
            "bert_prompt": "[MASK] works for an airline, just like a sailor works for a ship.",
            "distilbert_prompt": "[MASK] works for an airline, just like a sailor works for a ship.",
            "roberta_prompt": "<mask> works for an airline, just like a sailor works for a ship.",
            "t5_input": "fill mask: [MASK] works for an airline, just like a sailor works for a <extra_id_0>.",
            "gpt_prompt": "Works-for relation: ____ works for an airline, just like a sailor works for a ship.\nAnswer:",
            "correct_answers": ["pilot"]
        },
        {
            "id": "sem_beg_02",
            "description": "place-to-resident (beginning)",
            "bert_prompt": "[MASK] is the capital of France and well-known for the Eiffel Tower.",
            "distilbert_prompt": "[MASK] is the capital of France and well-known for the Eiffel Tower.",
            "roberta_prompt": "<mask> is the capital of France and well-known for the Eiffel Tower.",
            "t5_input": "fill mask: [MASK] is the capital of France and well-known for the Eiffel Tower.",
            "gpt_prompt": "Question: ____ is the capital of France and famous for the Eiffel Tower.\nAnswer:",
            "correct_answers": ["paris"]
        },
        {
            "id": "sem_beg_03",
            "description": "tool-to-user (beginning)",
            "bert_prompt": "[MASK] is used for writing, just like a phone is used for calling.",
            "distilbert_prompt": "[MASK] is used for writing, just like a phone is used for calling.",
            "roberta_prompt": "<mask> is used for writing, just like a phone is used for calling.",
            "t5_input": "fill mask: [MASK] is used for writing, just like a phone is used for calling.",
            "gpt_prompt": "Complete: ____ is used for writing, just like a phone is used for calling.\nAnswer:",
            "correct_answers": ["pen", "pencil"]
        }


]