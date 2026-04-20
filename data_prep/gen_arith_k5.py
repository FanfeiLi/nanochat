"""Generate synthetic K5 arithmetic conversations for mid-training / SFT.

Output: JSONL where each line is a list of {role, content} messages
(CustomJSON-compatible). Same format as gsm8k_bounded_k5.jsonl so it can be
passed to chat_sft via --extra-jsonl, OR wrapped in a Task and consumed by
a revived mid_train.py.

Operations & difficulty (K5 scope):
  - addition / subtraction:  1-5 digit operands (with carry/borrow)
  - multiplication:          up to 3-digit × 2-digit (some 3×3)
  - division:                up to 3 ÷ 2 digits (exact + with remainder)
  - fractions:               a/b op c/d, denominators ≤ 12 (add/sub/mul/simplify)
  - decimals:                1-2 decimal places, mostly mul/div by integers

Surface formats (rotated to avoid format collapse — no "####" markers):
  - plain:     "What is 47 + 83?" / "47 + 83 = 130"
  - instr:     "Compute 47 + 83." / "The answer is 130."
  - word:      "Forty-seven plus eighty-three?" / "...equals one hundred thirty."
  - cot:       multiplication step-by-step (47×80=3760, 47×3=141, sum=3901)
  - story:     "Tom has 8 apples, Max has 7. Total?" / "8 + 7 = 15."
  - frac:      "1/2 + 1/3?" / "= 3/6 + 2/6 = 5/6"
  - decimal:   "2.5 + 3.75?" / "= 6.25"

Default: 1.5M examples ≈ ~45M tokens (~30 tok/example avg).

Usage:
    python gen_arith_k5.py --num-examples 1500000 --out /fast/fli/synthetic_k5_arith.jsonl --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import random
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 12

# ---------------------------------------------------------------------------
# operation distribution (must sum to 1.0)
OP_WEIGHTS = {
    "add":      0.20,
    "sub":      0.18,
    "mul":      0.18,
    "div":      0.12,
    "fraction": 0.16,
    "decimal":  0.16,
}

# ---------------------------------------------------------------------------
# helpers

NAMES = ["Tom", "Max", "Sara", "Ana", "Sam", "Lily", "Ben", "Mia", "Leo", "Ivy",
         "Eli", "Zoe", "Nina", "Owen", "Ruby", "Jay", "Ella", "Finn", "Noah", "Eva",
         "Liam", "Emma", "Oliver", "Ava", "Aiden", "Maya", "Lucas", "Sofia", "Amir",
         "Priya", "Kai", "Luna", "Felix", "Iris", "Rafi", "Nora", "Jin", "Amara",
         "Theo", "Olive"]
OBJECTS = ["apples", "cookies", "stickers", "pencils", "marbles", "books",
           "candies", "cards", "rocks", "shells", "stars", "blocks", "balls",
           "toys", "flowers", "leaves", "coins", "beads", "tickets", "ribbons",
           "feathers", "buttons", "erasers"]
ACTIONS = [
    ("has", "and", "How many in total?", "+"),     # add
    ("has", "gives", "How many are left?", "-"),   # sub
    ("buys", "boxes of", "How many in total?", "*"), # mul (X buys N boxes of M items)
    ("shares", "into", "boxes equally. How many in each?", "/"),  # div
]

NUMBER_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
}


def num_to_words(n: int) -> str:
    if n < 0:
        return "negative " + num_to_words(-n)
    if n in NUMBER_WORDS:
        return NUMBER_WORDS[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return NUMBER_WORDS[tens * 10] + ("-" + NUMBER_WORDS[ones] if ones else "")
    if n < 1000:
        h, rest = divmod(n, 100)
        head = NUMBER_WORDS[h] + " hundred"
        return head + (" " + num_to_words(rest) if rest else "")
    if n < 1_000_000:
        thousands, rest = divmod(n, 1000)
        head = num_to_words(thousands) + " thousand"
        return head + (" " + num_to_words(rest) if rest else "")
    return str(n)  # fallback (won't be hit at K5 scale)


# ---------------------------------------------------------------------------
# operand sampling

def sample_int(min_d: int, max_d: int) -> int:
    """Sample an integer with digit count uniformly in [min_d, max_d]."""
    d = random.randint(min_d, max_d)
    if d == 1:
        return random.randint(1, 9)
    return random.randint(10 ** (d - 1), 10 ** d - 1)


def sample_addsub() -> tuple[int, int]:
    digits_a = random.choices([1, 2, 3, 4, 5], weights=[5, 25, 35, 25, 10])[0]
    digits_b = random.choices([1, 2, 3, 4, 5], weights=[5, 25, 35, 25, 10])[0]
    a = sample_int(digits_a, digits_a)
    b = sample_int(digits_b, digits_b)
    return a, b


def sample_mul() -> tuple[int, int]:
    # K5 scope: mostly 1×1 to 3×2, occasional 3×3
    pattern = random.choices(
        [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)],
        weights=[20, 20, 25, 15, 15, 5])[0]
    a = sample_int(pattern[0], pattern[0])
    b = sample_int(pattern[1], pattern[1])
    return a, b


def sample_div() -> tuple[int, int]:
    # Build dividend = quotient × divisor (+ remainder). Keep divisor 1-2 digits.
    divisor_digits = random.choices([1, 2], weights=[60, 40])[0]
    quotient_digits = random.choices([1, 2, 3], weights=[40, 40, 20])[0]
    divisor = sample_int(divisor_digits, divisor_digits)
    quotient = sample_int(quotient_digits, quotient_digits)
    has_remainder = random.random() < 0.5
    remainder = random.randint(1, max(1, divisor - 1)) if has_remainder else 0
    dividend = quotient * divisor + remainder
    return dividend, divisor


def sample_fraction() -> tuple[int, int, int, int, str]:
    """Returns (a, b, c, d, op) with denominators 2..12, numerators 1..(b-1)."""
    op = random.choice(["+", "-", "*"])
    b = random.randint(2, 12)
    d = random.randint(2, 12)
    a = random.randint(1, b - 1)
    c = random.randint(1, d - 1)
    return a, b, c, d, op


def sample_decimal() -> tuple[Decimal, Decimal, str]:
    """Generate a clean K5 decimal expression."""
    op = random.choice(["+", "+", "-", "-", "*", "/"])
    if op in ("+", "-"):
        # both 1- or 2-decimal-place values
        dp_a = random.choice([1, 2])
        dp_b = random.choice([1, 2])
        a = Decimal(random.randint(0, 99 if dp_a == 2 else 9)) / Decimal(10 ** dp_a)
        a += Decimal(random.randint(0, 99))
        b = Decimal(random.randint(0, 99 if dp_b == 2 else 9)) / Decimal(10 ** dp_b)
        b += Decimal(random.randint(0, 99))
        # for sub, ensure a >= b for clean K5 output
        if op == "-" and a < b:
            a, b = b, a
        return a.quantize(Decimal("0.01")), b.quantize(Decimal("0.01")), op
    if op == "*":
        # decimal × small integer
        dp = random.choice([1, 2])
        a = Decimal(random.randint(1, 99)) / Decimal(10 ** dp)
        a += Decimal(random.randint(0, 9))
        b = Decimal(random.randint(2, 12))
        return a.quantize(Decimal("0.01")), b, op
    # op == "/": clean integer-quotient division of decimal by small int
    b = Decimal(random.randint(2, 9))
    quotient = Decimal(random.randint(1, 25))
    quotient = quotient / Decimal(random.choice([10, 100]))
    a = (quotient * b).quantize(Decimal("0.01"))
    return a, b, op


# ---------------------------------------------------------------------------
# format renderers
# Each returns (user_str, assistant_str). Assistant always shows the inline
# arithmetic (no "####" marker, no calculator tags).

OP_SYMBOLS = {"add": "+", "sub": "-", "mul": "×", "div": "÷"}
OP_VERBS = {"add": "plus", "sub": "minus", "mul": "times", "div": "divided by"}


def fmt_plain_int(a: int, b: int, op_name: str) -> tuple[str, str]:
    sym = OP_SYMBOLS[op_name]
    if op_name == "div":
        q, r = divmod(a, b)
        u = f"What is {a} {sym} {b}?"
        if r == 0:
            return u, f"{a} {sym} {b} = {q}"
        return u, f"{a} {sym} {b} = {q} remainder {r}"
    res = {"add": a + b, "sub": a - b, "mul": a * b}[op_name]
    return f"What is {a} {sym} {b}?", f"{a} {sym} {b} = {res}"


def fmt_instr_int(a: int, b: int, op_name: str) -> tuple[str, str]:
    sym = OP_SYMBOLS[op_name]
    verb = {"add": "Add", "sub": "Subtract", "mul": "Multiply", "div": "Divide"}[op_name]
    if op_name == "div":
        q, r = divmod(a, b)
        u = f"{verb} {a} {'by' if op_name in ('div',) else 'and'} {b}."
        if r == 0:
            return u, f"The answer is {q}."
        return u, f"{a} {sym} {b} = {q} with remainder {r}."
    res = {"add": a + b, "sub": a - b, "mul": a * b}[op_name]
    u = f"{verb} {a} and {b}."
    return u, f"The answer is {res}."


def fmt_word_int(a: int, b: int, op_name: str) -> tuple[str, str]:
    if max(a, b) > 999_999:  # word form gets unwieldy
        return fmt_plain_int(a, b, op_name)
    verb = OP_VERBS[op_name]
    aw, bw = num_to_words(a), num_to_words(b)
    if op_name == "div":
        q, r = divmod(a, b)
        if r == 0:
            return (f"What is {aw} {verb} {bw}?",
                    f"{aw.capitalize()} {verb} {bw} equals {num_to_words(q)}.")
        return fmt_plain_int(a, b, op_name)
    res = {"add": a + b, "sub": a - b, "mul": a * b}[op_name]
    return (f"What is {aw} {verb} {bw}?",
            f"{aw.capitalize()} {verb} {bw} equals {num_to_words(res)}.")


def fmt_cot_mul(a: int, b: int) -> tuple[str, str]:
    """Step-by-step multiplication: split smaller operand into place values."""
    if b > a:
        a, b = b, a  # ensure a >= b for cleaner steps
    if b < 10:
        return fmt_plain_int(a, b, "mul")
    digits = [int(c) for c in str(b)]
    parts = []
    running = 0
    for i, d in enumerate(digits):
        place = 10 ** (len(digits) - 1 - i)
        chunk = a * d * place
        if d != 0:
            parts.append(f"{a} × {d * place} = {chunk}")
        running += chunk
    final = a * b
    steps = "; ".join(parts)
    return (f"Multiply {a} × {b}.",
            f"{steps}; sum = {final}. So {a} × {b} = {final}.")


def _int_add_templates(n1: str, n2: str, a: int, b: int, obj: str, s: int) -> list[tuple[str, str]]:
    return [
        (f"{n1} has {a} {obj}. {n2} has {b} {obj}. How many do they have together?",
         f"{a} + {b} = {s}. They have {s} {obj} together."),
        (f"{n1} collected {a} {obj} in the morning and {b} more in the afternoon. How many in total?",
         f"{a} + {b} = {s}. {n1} collected {s} {obj} in total."),
        (f"There were {a} {obj} in the box. {n1} added {b} more. How many now?",
         f"{a} + {b} = {s}. Now there are {s} {obj}."),
        (f"{n1} started with {a} {obj} and earned {b} more. How many does {n1} have now?",
         f"{a} + {b} = {s}. {n1} has {s} {obj} now."),
        (f"On Monday there were {a} {obj}. On Tuesday {b} more arrived. What is the total?",
         f"{a} + {b} = {s}. The total is {s} {obj}."),
        (f"{n1} counted {a} {obj} at home and {b} more at school. How many altogether?",
         f"{a} + {b} = {s}. Altogether there are {s} {obj}."),
        (f"A jar had {a} {obj} and {n1} added {b} more. How many are in the jar now?",
         f"{a} + {b} = {s}. The jar now has {s} {obj}."),
        (f"{n1} found {a} {obj} and {n2} found {b} {obj}. How many in total?",
         f"{a} + {b} = {s}. They found {s} {obj} in total."),
    ]


def _int_sub_templates(n1: str, n2: str, a: int, b: int, obj: str, d: int) -> list[tuple[str, str]]:
    return [
        (f"{n1} has {a} {obj}. {n1} gives {b} to {n2}. How many does {n1} have left?",
         f"{a} - {b} = {d}. {n1} has {d} {obj} left."),
        (f"{n1} had {a} {obj} and lost {b}. How many remain?",
         f"{a} - {b} = {d}. {n1} has {d} {obj} remaining."),
        (f"There were {a} {obj} in the jar. {n1} took out {b}. How many are still in the jar?",
         f"{a} - {b} = {d}. There are {d} {obj} still in the jar."),
        (f"{n1} started with {a} {obj} and used {b}. How many are left?",
         f"{a} - {b} = {d}. {n1} has {d} {obj} left."),
        (f"A store had {a} {obj} and sold {b}. How many remain in the store?",
         f"{a} - {b} = {d}. The store has {d} {obj} remaining."),
        (f"{n1} collected {a} {obj} and gave away {b} to {n2}. How many does {n1} keep?",
         f"{a} - {b} = {d}. {n1} keeps {d} {obj}."),
        (f"Out of {a} {obj}, {b} are broken. How many are still good?",
         f"{a} - {b} = {d}. {d} {obj} are still good."),
        (f"{n1} had {a} {obj} yesterday and has {b} fewer today. How many does {n1} have today?",
         f"{a} - {b} = {d}. {n1} has {d} {obj} today."),
    ]


def _int_mul_templates(n1: str, n2: str, per_group: int, num_groups: int, obj: str, p: int) -> list[tuple[str, str]]:
    a, b = per_group, num_groups
    return [
        (f"{n1} buys {b} boxes of {obj}. Each box has {a} {obj}. How many {obj} in total?",
         f"{a} × {b} = {p}. There are {p} {obj} in total."),
        (f"{n1} plants {b} rows with {a} flowers per row. How many flowers in total?",
         f"{a} × {b} = {p}. There are {p} flowers."),
        (f"A bag holds {a} {obj}. {n1} has {b} bags. How many {obj} in all?",
         f"{a} × {b} = {p}. There are {p} {obj} in all."),
        (f"{n1} makes {b} batches of cookies with {a} cookies per batch. How many cookies?",
         f"{a} × {b} = {p}. {n1} makes {p} cookies."),
        (f"Each shelf has {a} {obj} and there are {b} shelves. How many {obj} in total?",
         f"{a} × {b} = {p}. There are {p} {obj} in total."),
        (f"{n1} reads {a} pages per day for {b} days. How many pages does {n1} read?",
         f"{a} × {b} = {p}. {n1} reads {p} pages."),
        (f"A box contains {a} {obj}. {n2} buys {b} boxes. How many {obj} does {n2} have?",
         f"{a} × {b} = {p}. {n2} has {p} {obj}."),
        (f"{n1} walks {a} blocks each day for {b} days. How many blocks total?",
         f"{a} × {b} = {p}. {n1} walks {p} blocks total."),
    ]


def _int_div_templates(n1: str, n2: str, a: int, b: int, obj: str, q: int) -> list[tuple[str, str]]:
    return [
        (f"{n1} has {a} {obj} and shares them equally into {b} bags. How many in each bag?",
         f"{a} ÷ {b} = {q}. Each bag has {q} {obj}."),
        (f"{n1} divides {a} {obj} among {b} friends equally. How many {obj} per friend?",
         f"{a} ÷ {b} = {q}. Each friend gets {q} {obj}."),
        (f"{a} {obj} are packed into boxes of {b}. How many boxes are needed?",
         f"{a} ÷ {b} = {q}. {q} boxes are needed."),
        (f"A teacher has {a} pencils to share equally among {b} students. How many pencils per student?",
         f"{a} ÷ {b} = {q}. Each student gets {q} pencils."),
        (f"{n1} arranges {a} {obj} into {b} equal rows. How many {obj} per row?",
         f"{a} ÷ {b} = {q}. Each row has {q} {obj}."),
        (f"A farmer has {a} {obj} and puts them in {b} equal piles. How many {obj} per pile?",
         f"{a} ÷ {b} = {q}. Each pile has {q} {obj}."),
        (f"If {a} {obj} are split equally between {b} kids, how many does each kid get?",
         f"{a} ÷ {b} = {q}. Each kid gets {q} {obj}."),
    ]


def fmt_story(a: int, b: int, op_name: str) -> tuple[str, str]:
    n1, n2 = random.sample(NAMES, 2)
    obj = random.choice(OBJECTS)
    if op_name == "add":
        templates = _int_add_templates(n1, n2, a, b, obj, a + b)
    elif op_name == "sub":
        if a < b:
            a, b = b, a
        templates = _int_sub_templates(n1, n2, a, b, obj, a - b)
    elif op_name == "mul":
        templates = _int_mul_templates(n1, n2, a, b, obj, a * b)
    else:  # "div"
        q, r = divmod(a, b)
        if r != 0:
            return fmt_plain_int(a, b, op_name)
        templates = _int_div_templates(n1, n2, a, b, obj, q)
    return random.choice(templates)


FRAC_NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                  6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
                  11: "eleven", 12: "twelve"}
FRAC_DEN_WORDS = {2: "half", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
                  7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
                  11: "eleventh", 12: "twelfth"}
FRAC_OBJECTS = ["pizza", "cake", "chocolate bar", "apple pie", "sandwich",
                "water tank", "bag of flour", "ribbon", "garden", "bowl of fruit",
                "jar of cookies", "rope", "carton of juice", "loaf of bread",
                "pan of brownies", "bottle of water", "spool of thread"]


def _a_or_an(noun: str) -> str:
    """Return 'a' or 'an' based on the first letter of the noun."""
    return "an" if noun and noun[0].lower() in "aeiou" else "a"


def _frac_to_words(n: int, d: int) -> str:
    """1/2 -> 'one half'; 3/4 -> 'three fourths'; 1/12 -> 'one twelfth'."""
    num_w = FRAC_NUM_WORDS.get(n, str(n))
    den_w = FRAC_DEN_WORDS.get(d, f"{d}th")
    if n > 1:
        den_w += "s"
    return f"{num_w} {den_w}"


def _frac_compute(a: int, b: int, c: int, d: int, op: str) -> tuple[int, int, int, int, int, int]:
    """Return (lcd, a2, c2, num, den_final, gcd_simplify)."""
    lcd = (b * d) // math.gcd(b, d)
    a2 = a * (lcd // b)
    c2 = c * (lcd // d)
    if op == "+":
        num = a2 + c2; den = lcd
    elif op == "-":
        num = a2 - c2; den = lcd
    else:  # "*"
        num = a * c; den = b * d
    g = math.gcd(abs(num) if num else 1, den)
    return lcd, a2, c2, num, den, g


def fmt_fraction_plain(a: int, b: int, c: int, d: int, op: str) -> tuple[str, str]:
    """Compact plain-style with LCD conversion step (original format)."""
    if op == "-":
        lcd = (b * d) // math.gcd(b, d)
        a2 = a * (lcd // b); c2 = c * (lcd // d)
        if a2 < c2:
            a, b, c, d = c, d, a, b
    lcd, a2, c2, num, den, g = _frac_compute(a, b, c, d, op)
    sym = {"+": "+", "-": "-", "*": "×"}[op]
    u = f"What is {a}/{b} {sym} {c}/{d}?"
    if op == "*":
        v = f"{a}/{b} × {c}/{d} = {num}/{den}"
    elif b == d:
        v = f"{a}/{b} {sym} {c}/{d} = {num}/{lcd}"
    else:
        v = (f"{a}/{b} = {a2}/{lcd} and {c}/{d} = {c2}/{lcd}, "
             f"so {a}/{b} {sym} {c}/{d} = {a2}/{lcd} {sym} {c2}/{lcd} = {num}/{lcd}")
    if g > 1 and op != "*":
        v += f" = {num // g}/{lcd // g}"
    elif g > 1 and op == "*":
        v += f" = {num // g}/{den // g}"
    return u, v


def fmt_fraction_cot(a: int, b: int, c: int, d: int, op: str) -> tuple[str, str]:
    """Explicit step-by-step explanation (pedagogical)."""
    if op == "-":
        lcd = (b * d) // math.gcd(b, d)
        a2 = a * (lcd // b); c2 = c * (lcd // d)
        if a2 < c2:
            a, b, c, d = c, d, a, b
    lcd, a2, c2, num, den, g = _frac_compute(a, b, c, d, op)
    sym = {"+": "+", "-": "-", "*": "×"}[op]
    u = f"Compute {a}/{b} {sym} {c}/{d}. Show your work."
    if op == "*":
        steps = [
            f"Multiply the numerators: {a} × {c} = {num}.",
            f"Multiply the denominators: {b} × {d} = {den}.",
            f"So {a}/{b} × {c}/{d} = {num}/{den}.",
        ]
        if g > 1:
            steps.append(f"Simplify by dividing by {g}: {num // g}/{den // g}.")
        return u, " ".join(steps)
    # + or -
    steps = []
    if b == d:
        steps.append(f"The denominators are the same ({b}), so we can {'add' if op == '+' else 'subtract'} the numerators directly.")
        steps.append(f"{a} {sym} {c} = {num}.")
        steps.append(f"So {a}/{b} {sym} {c}/{d} = {num}/{lcd}.")
    else:
        steps.append(f"First find a common denominator. The LCD of {b} and {d} is {lcd}.")
        steps.append(f"Convert: {a}/{b} = {a2}/{lcd} (multiply top and bottom by {lcd // b}).")
        steps.append(f"Convert: {c}/{d} = {c2}/{lcd} (multiply top and bottom by {lcd // d}).")
        steps.append(f"Now {'add' if op == '+' else 'subtract'}: {a2}/{lcd} {sym} {c2}/{lcd} = {num}/{lcd}.")
    if g > 1:
        steps.append(f"Simplify by dividing by {g}: {num // g}/{lcd // g}.")
    return u, " ".join(steps)


def fmt_fraction_word(a: int, b: int, c: int, d: int, op: str) -> tuple[str, str]:
    """Pure word form: 'what is one half plus one third?'."""
    if op == "-":
        lcd = (b * d) // math.gcd(b, d)
        a2 = a * (lcd // b); c2 = c * (lcd // d)
        if a2 < c2:
            a, b, c, d = c, d, a, b
    lcd, a2, c2, num, den, g = _frac_compute(a, b, c, d, op)
    verb = {"+": "plus", "-": "minus", "*": "times"}[op]
    aw = _frac_to_words(a, b)
    cw = _frac_to_words(c, d)
    u = f"What is {aw} {verb} {cw}?"
    # Simplify result for cleaner word expression
    if op == "*":
        rn, rd = num, den
    else:
        rn, rd = num, lcd
    if g > 1:
        rn //= g; rd //= g
    rw = _frac_to_words(rn, rd) if 1 <= rn and rd in FRAC_DEN_WORDS else f"{rn}/{rd}"
    v = f"{aw.capitalize()} {verb} {cw} equals {rw}."
    return u, v


def fmt_fraction_story(a: int, b: int, c: int, d: int, op: str) -> tuple[str, str]:
    """Story form. Most natural for + and -. Falls back for * since frac-of-frac is awkward."""
    if op == "*":
        return fmt_fraction_plain(a, b, c, d, op)
    if op == "-":
        lcd = (b * d) // math.gcd(b, d)
        a2 = a * (lcd // b); c2 = c * (lcd // d)
        if a2 < c2:
            a, b, c, d = c, d, a, b
    lcd, a2, c2, num, den, g = _frac_compute(a, b, c, d, op)
    sym = {"+": "+", "-": "-"}[op]
    n1, n2 = random.sample(NAMES, 2)
    obj = random.choice(FRAC_OBJECTS)
    art = _a_or_an(obj)
    if b == d:
        calc_add = f"{a}/{b} + {c}/{d} = {num}/{lcd}"
        calc_sub = f"{a}/{b} - {c}/{d} = {num}/{lcd}"
    else:
        calc_add = (f"{a}/{b} = {a2}/{lcd} and {c}/{d} = {c2}/{lcd}, "
                    f"so {a}/{b} + {c}/{d} = {a2}/{lcd} + {c2}/{lcd} = {num}/{lcd}")
        calc_sub = (f"{a}/{b} = {a2}/{lcd} and {c}/{d} = {c2}/{lcd}, "
                    f"so {a}/{b} - {c}/{d} = {a2}/{lcd} - {c2}/{lcd} = {num}/{lcd}")
    result_str = f"{num}/{lcd}" if g == 1 else f"{num // g}/{lcd // g}"

    if op == "+":
        templates = [
            (f"{n1} ate {a}/{b} of {art} {obj}. {n2} ate {c}/{d} of the same kind of {obj}. How much did they eat together?",
             f"{calc_add}. Together they ate {result_str} of {art} {obj}."),
            (f"{n1} drank {a}/{b} of {art} {obj} in the morning and {c}/{d} of {art} {obj} in the afternoon. How much did {n1} drink in total?",
             f"{calc_add}. {n1} drank {result_str} of {art} {obj} in total."),
            (f"In the first week, {n1} painted {a}/{b} of {art} {obj}. In the second week, {n1} painted {c}/{d} more of {art} {obj}. How much is painted in total?",
             f"{calc_add}. {n1} painted {result_str} of {art} {obj} in total."),
            (f"A recipe needs {a}/{b} cup of sugar and {c}/{d} cup more. How much sugar is needed in total?",
             f"{calc_add}. The recipe needs {result_str} cups of sugar in total."),
            (f"{n1} read {a}/{b} of a book yesterday and {c}/{d} of the book today. How much has {n1} read in total?",
             f"{calc_add}. {n1} has read {result_str} of the book in total."),
            (f"{n1} used {a}/{b} of {art} {obj} and {n2} used {c}/{d} of {art} {obj}. How much did they use together?",
             f"{calc_add}. Together they used {result_str} of {art} {obj}."),
        ]
    else:  # op == "-"
        templates = [
            (f"{n1} had {a}/{b} of {art} {obj} and gave {c}/{d} of {art} {obj} to {n2}. How much did {n1} have left?",
             f"{calc_sub}. {n1} has {result_str} of {art} {obj} left."),
            (f"{n1} started with {a}/{b} of {art} {obj} and ate {c}/{d} of {art} {obj}. How much is left?",
             f"{calc_sub}. There is {result_str} of {art} {obj} left."),
            (f"A jar was {a}/{b} full. {n1} used {c}/{d} of the jar. How full is the jar now?",
             f"{calc_sub}. The jar is now {result_str} full."),
            (f"{n1} owned {a}/{b} of {art} {obj} and sold {c}/{d} of {art} {obj}. How much does {n1} have remaining?",
             f"{calc_sub}. {n1} has {result_str} of {art} {obj} remaining."),
            (f"A tank was {a}/{b} full of water. {c}/{d} of the tank was drained. How much water is left?",
             f"{calc_sub}. The tank is {result_str} full."),
            (f"{n1} had {a}/{b} of {art} {obj}. After sharing {c}/{d} of {art} {obj} with friends, how much does {n1} have left?",
             f"{calc_sub}. {n1} has {result_str} of {art} {obj} left."),
        ]
    return random.choice(templates)


def fmt_fraction_short(a: int, b: int, c: int, d: int, op: str) -> tuple[str, str]:
    """Minimal Q/A — just the final answer (good for fast-recall practice)."""
    if op == "-":
        lcd = (b * d) // math.gcd(b, d)
        a2 = a * (lcd // b); c2 = c * (lcd // d)
        if a2 < c2:
            a, b, c, d = c, d, a, b
    lcd, a2, c2, num, den, g = _frac_compute(a, b, c, d, op)
    sym = {"+": "+", "-": "-", "*": "×"}[op]
    u = f"{a}/{b} {sym} {c}/{d} = ?"
    final_den = den if op == "*" else lcd
    if g > 1:
        v = f"{num // g}/{final_den // g}"
    else:
        v = f"{num}/{final_den}"
    return u, v


FRAC_FORMATS = [
    ("plain", fmt_fraction_plain, 0.35),
    ("cot",   fmt_fraction_cot,   0.20),
    ("word",  fmt_fraction_word,  0.15),
    ("story", fmt_fraction_story, 0.20),
    ("short", fmt_fraction_short, 0.10),
]


def fmt_fraction(a: int, b: int, c: int, d: int, op: str) -> tuple[str, str]:
    """Dispatch to one of several fraction surface formats."""
    names, fns, weights = zip(*FRAC_FORMATS)
    fn = random.choices(fns, weights=weights)[0]
    return fn(a, b, c, d, op)


def fmt_decimal(a: Decimal, b: Decimal, op: str) -> tuple[str, str]:
    sym_map = {"+": "+", "-": "-", "*": "×", "/": "÷"}
    sym = sym_map[op]
    if op == "+":
        res = a + b
    elif op == "-":
        res = a - b
    elif op == "*":
        res = a * b
    else:
        res = a / b
    res = res.quantize(Decimal("0.01"))
    # trim trailing zero past the decimal point if it's clean (e.g. 6.50 -> 6.5)
    res_str = format(res.normalize(), "f")
    if "." in res_str and res_str.endswith("0"):
        res_str = res_str.rstrip("0").rstrip(".")
    a_str = format(a, "f")
    b_str = format(b, "f")
    return f"What is {a_str} {sym} {b_str}?", f"{a_str} {sym} {b_str} = {res_str}"


# ---------------------------------------------------------------------------
# top-level example builder

INT_FORMATS = ["plain", "instr", "word", "story"]


def build_example(rng: random.Random) -> list[dict]:
    op_name = rng.choices(list(OP_WEIGHTS.keys()), weights=list(OP_WEIGHTS.values()))[0]

    if op_name == "fraction":
        a, b, c, d, op = sample_fraction()
        u, v = fmt_fraction(a, b, c, d, op)
    elif op_name == "decimal":
        a, b, op = sample_decimal()
        u, v = fmt_decimal(a, b, op)
    else:
        # integer ops: choose format by weighted slot
        if op_name == "add":
            a, b = sample_addsub()
        elif op_name == "sub":
            a, b = sample_addsub()
            if a < b:
                a, b = b, a
        elif op_name == "mul":
            a, b = sample_mul()
        elif op_name == "div":
            a, b = sample_div()
        # Choose surface format
        # multiplication CoT only for non-trivial cases to teach long-mult
        if op_name == "mul" and rng.random() < 0.30 and (a >= 10 or b >= 10):
            u, v = fmt_cot_mul(a, b)
        else:
            fmt = rng.choices(INT_FORMATS, weights=[40, 20, 20, 20])[0]
            if fmt == "plain":
                u, v = fmt_plain_int(a, b, op_name)
            elif fmt == "instr":
                u, v = fmt_instr_int(a, b, op_name)
            elif fmt == "word":
                u, v = fmt_word_int(a, b, op_name)
            else:  # story
                u, v = fmt_story(a, b, op_name)

    return [
        {"role": "user", "content": u},
        {"role": "assistant", "content": v},
    ]


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-examples", type=int, default=1_500_000,
                   help="number of conversations to generate (default 1.5M ~= 45M tokens)")
    p.add_argument("--out", type=str, default="/fast/fli/synthetic_k5_arith.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample", action="store_true",
                   help="just print 10 examples and exit (smoke test)")
    args = p.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)

    if args.sample:
        for i in range(15):
            ex = build_example(rng)
            print(f"--- example {i} ---")
            print(f"  USER:      {ex[0]['content']}")
            print(f"  ASSISTANT: {ex[1]['content']}")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating {args.num_examples:,} arithmetic examples -> {out_path}")
    n_chars = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(args.num_examples):
            ex = build_example(rng)
            line = json.dumps(ex, ensure_ascii=False)
            f.write(line + "\n")
            n_chars += len(line)
            if (i + 1) % 100_000 == 0:
                approx_tok = n_chars // 4
                print(f"  {i+1:,}/{args.num_examples:,}  "
                      f"approx_tokens≈{approx_tok:,}  ({approx_tok / 1e6:.1f} M)")
    approx_tok = n_chars // 4
    print(f"Done. Wrote {args.num_examples:,} examples, ~{approx_tok:,} tokens "
          f"({approx_tok / 1e6:.1f} M).")
    print(f"File size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
