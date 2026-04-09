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
        gpt2_prompt=f"A {a} works for a {a_place}. Similarly, a {b} is worked at by a"
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

# ============================================================
# Taxonomic analogies — FIXED T5 PROMPTS
# ============================================================
taxonomic_examples = [
    ("sparrow", "bird", "salmon", ["fish"]),
    ("lion", "mammal", "eagle", ["bird"]),
    ("rose", "flower", "oak", ["tree"]),
    ("whale", "mammal", "shark", ["fish"]),
    ("frog", "amphibian", "toad", ["amphibian"]),
    ("cucumber", "vegetable", "apple", ["fruit"]),
    ("python", "snake", "cobra", ["snake"]),
    ("tiger", "mammal", "cheetah", ["mammal"]),
    ("daffodil", "flower", "tulip", ["flower"]),
]

for idx, (a, a_class, b, answers) in enumerate(taxonomic_examples, 1):

    add_analogy(
        id=f"taxonomic_{idx:03d}_beginning",
        analogy_type="taxonomic",
        mask_position="beginning",
        correct_answers=[a],
        bert_prompt=f"[MASK] is a {a_class}, just like a {b} is a {answers[0]}.",
        distil_prompt=f"[MASK] is a {a_class}, just like a {b} is a {answers[0]}.",
        roberta_prompt=f"<mask> is a {a_class}, just like a {b} is a {answers[0]}.",
        t5_prompt=f"fill: <extra_id_0> is a {a_class}, just like a {b} is a {answers[0]}.",
        gpt2_prompt=f"{b.capitalize()} is a {answers[0]}. Similarly, a {a_class} would include a"
    )

    add_analogy(
        id=f"taxonomic_{idx:03d}_middle",
        analogy_type="taxonomic",
        mask_position="middle",
        correct_answers=[a_class],
        bert_prompt=f"A {a} is a [MASK], just like a {b} is a {answers[0]}.",
        distil_prompt=f"A {a} is a [MASK], just like a {b} is a {answers[0]}.",
        roberta_prompt=f"A {a} is a <mask>, just like a {b} is a {answers[0]}.",
        t5_prompt=f"fill: A {a} is a <extra_id_0>, just like a {b} is a {answers[0]}.",
        gpt2_prompt=f"A {a} is a {a_class}. Similarly, a {answers[0]} would include a"
    )

    add_analogy(
        id=f"taxonomic_{idx:03d}_end",
        analogy_type="taxonomic",
        mask_position="end",
        correct_answers=answers,
        bert_prompt=f"A {a} is a {a_class}, just like a {b} is a [MASK].",
        distil_prompt=f"A {a} is a {a_class}, just like a {b} is a [MASK].",
        roberta_prompt=f"A {a} is a {a_class}, just like a {b} is a <mask>.",
        t5_prompt=f"fill: A {a} is a {a_class}, just like a {b} is a <extra_id_0>.",
        gpt2_prompt=f"A {a} is a {a_class}. Similarly, a {b} is a"
    )

# ============================================================
# Role-object analogies — FIXED T5 PROMPTS
# ============================================================
role_object_examples = [
    ("chef", "kitchen", "teacher", ["school", "classroom"]),
    ("pilot", "plane", "driver", ["car"]),
    ("writer", "office", "artist", ["studio"]),
    ("doctor", "hospital", "nurse", ["clinic"]),
    ("musician", "instrument", "singer", ["microphone", "stage"]),
    ("farmer", "field", "fisherman", ["boat", "river"]),
    ("judge", "court", "lawyer", ["courtroom", "chamber"]),
    ("coach", "team", "teacher", ["classroom", "students"]),
    ("scientist", "lab", "engineer", ["lab", "workshop"]),
    ("driver", "car", "bus_driver", ["bus"])
]

for idx, (role, role_place, other_role, answers) in enumerate(role_object_examples, 1):

    add_analogy(
        id=f"role_{idx:03d}_beginning",
        analogy_type="role-object",
        mask_position="beginning",
        correct_answers=[role],
        bert_prompt=f"[MASK] works in a {role_place}, just like a {other_role} works in a {answers[0]}.",
        distil_prompt=f"[MASK] works in a {role_place}, just like a {other_role} works in a {answers[0]}.",
        roberta_prompt=f"<mask> works in a {role_place}, just like a {other_role} works in a {answers[0]}.",
        t5_prompt=f"fill: <extra_id_0> works in a {role_place}, just like a {other_role} works in a {answers[0]}.",
        gpt2_prompt=f"{other_role} works in a {answers[0]}. Similarly, a {answers[0]} is worked at by a"
    )

    add_analogy(
        id=f"role_{idx:03d}_middle",
        analogy_type="role-object",
        mask_position="middle",
        correct_answers=[role_place],
        bert_prompt=f"A {role} works in a [MASK], just like a {other_role} works in a {answers[0]}.",
        distil_prompt=f"A {role} works in a [MASK], just like a {other_role} works in a {answers[0]}.",
        roberta_prompt=f"A {role} works in a <mask>, just like a {other_role} works in a {answers[0]}.",
        t5_prompt=f"fill: A {role} works in a <extra_id_0>, just like a {other_role} works in a {answers[0]}.",
        gpt2_prompt=f"A {other_role} works in a {answers[0]}. Similarly, a {role} works in a"
    )

    add_analogy(
        id=f"role_{idx:03d}_end",
        analogy_type="role-object",
        mask_position="end",
        correct_answers=answers,
        bert_prompt=f"A {role} works in a {role_place}, just like a {other_role} works in a [MASK].",
        distil_prompt=f"A {role} works in a {role_place}, just like a {other_role} works in a [MASK].",
        roberta_prompt=f"A {role} works in a {role_place}, just like a {other_role} works in a <mask>.",
        t5_prompt=f"fill: A {role} works in a {role_place}, just like a {other_role} works in a <extra_id_0>.",
        gpt2_prompt=f"A {role} works in a {role_place}. Similarly, a {other_role} works in a"
    )

print(f"✅ Medium benchmark created with {len(benchmark)} examples.")
