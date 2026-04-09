# ============================================================
# T5 PROMPT FIX:
# Replace ALL "[MASK]" with "<extra_id_0>"
# Remove "fill mask:" → Replace with "fill: "
# ============================================================

benchmark = []

def add_analogy(id, analogy_type, mask_position, correct_answers,
                bert_prompt, distil_prompt, roberta_prompt, t5_prompt, gpt2_prompt):
    benchmark.append({
        "id": id,
        "analogy_type": analogy_type,
        "mask_position": mask_position,
        "correct_answers": correct_answers,
        "bert_prompt": bert_prompt,
        "distilbert_prompt": distil_prompt,
        "roberta_prompt": roberta_prompt,
        "t5_prompt": t5_prompt,
        "gpt2_prompt": gpt2_prompt
    })

# ============================================================
# Functional analogies — FIXED T5 PROMPTS
# ============================================================
functional_examples = [
    ("pilot", "airline", "sailor", ["ship", "boat", "vessel"]),
    ("chef", "kitchen", "teacher", ["school", "classroom"]),
    ("driver", "car", "captain", ["ship", "boat"]),
    ("writer", "pen", "artist", ["brush", "paint"]),
    ("farmer", "field", "fisherman", ["river", "boat"]),
    ("doctor", "hospital", "nurse", ["hospital", "clinic"]),
    ("judge", "court", "lawyer", ["courtroom", "chamber"]),
    ("pilot", "plane", "astronaut", ["spaceship", "rocket"]),
    ("musician", "instrument", "singer", ["microphone", "stage"]),
    ("coach", "team", "teacher", ["class", "students"])
]

for idx, (a, a_place, b, answers) in enumerate(functional_examples, 1):

    add_analogy(
        id=f"functional_{idx:03d}_beginning",
        analogy_type="functional",
        mask_position="beginning",
        correct_answers=[a],
        bert_prompt=f"[MASK] works for a {a_place}, just like a {b} works for a {answers[0]}.",
        distil_prompt=f"[MASK] works for a {a_place}, just like a {b} works for a {answers[0]}.",
        roberta_prompt=f"<mask> works for a {a_place}, just like a {b} works for a {answers[0]}.",
        t5_prompt=f"fill: <extra_id_0> works for a {a_place}, just like a {b} works for a {answers[0]}.",
        gpt2_prompt=f"{b.capitalize()} works for a {answers[0]}. Similarly, a {a_place} is worked at by a"
    )

    add_analogy(
        id=f"functional_{idx:03d}_middle",
        analogy_type="functional",
        mask_position="middle",
        correct_answers=[a_place],
        bert_prompt=f"A {a} works for a [MASK], just like a {b} works for a {answers[0]}.",
        distil_prompt=f"A {a} works for a [MASK], just like a {b} works for a {answers[0]}.",
        roberta_prompt=f"A {a} works for a <mask>, just like a {b} works for a {answers[0]}.",
        t5_prompt=f"fill: A {a} works for a <extra_id_0>, just like a {b} works for a {answers[0]}.",
        gpt2_prompt=f"A {a} works for a {a_place}. Similarly, a {answers[0]} is worked at by a"
    )

    add_analogy(
        id=f"functional_{idx:03d}_end",
        analogy_type="functional",
        mask_position="end",
        correct_answers=answers,
        bert_prompt=f"A {a} works for a {a_place}, just like a {b} works for a [MASK].",
        distil_prompt=f"A {a} works for a {a_place}, just like a {b} works for a [MASK].",
        roberta_prompt=f"A {a} works for a {a_place}, just like a {b} works for a <mask>.",
        t5_prompt=f"fill: A {a} works for a {a_place}, just like a {b} works for a <extra_id_0>.",
        gpt2_prompt=f"A {a} works for a {a_place}. Similarly, a {b} works for a"
    )