"""Conservative, evaluation-only answer comparison. Never execute model text.

Parse the complete expression, not an extractable substring. Preserve ordered
tuples and compare the values of numeric equalities, not their truth values.
Unsupported notation remains unknown. This is not a reasoning verifier.
"""

import re

VERSION = 'typed-local-math-v2'


class UnsupportedAnswer(ValueError):
    pass


def split_top(text, separator=','):
    depth, start, parts = 0, 0, []
    for i, char in enumerate(text):
        if char in '({[':
            depth += 1
        elif char in ')}]':
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def clean(text):
    text = text.replace(r'\left', '').replace(r'\right', '').replace(r'\displaystyle', '')
    text = text.replace(r'\$', '').replace('$', '').strip()
    text = text.replace(r'\,', ' ').replace(r'\!', ' ').replace(r'\;', ' ')
    # Avoid the converter reading d\cdot as a differential named 'dcdot'.
    return re.sub(r'\\(?:cdot|times)\b', '*', text)


def parse_math(text, task=''):
    import sympy as s
    from latex2sympy2_extended import latex2sympy
    from latex2sympy2_extended.latex2sympy2 import ConversionConfig
    from latex2sympy2_extended.math_normalization import NormalizationConfig

    if clean(text).lower() in {'true', 'false', 'unknown', 'none'}:
        raise UnsupportedAnswer('Not a mathematical expression')
    value = latex2sympy(clean(text), conversion_config=ConversionConfig(lowercase_symbols=False),
        normalization_config=NormalizationConfig(units=False, malformed_operators=True, boxed='none'))
    if isinstance(value, (s.logic.boolalg.BooleanTrue, s.logic.boolalg.BooleanFalse)):
        raise UnsupportedAnswer('A Boolean truth value is not a mathematical answer')
    # TeX juxtaposition of explicitly mentioned one-letter variables is a product.
    declared = set(re.findall(r'(?<![A-Za-z\\])[A-Za-z](?![A-Za-z])', task))
    substitutions = {}
    for symbol in value.free_symbols:
        name = str(symbol)
        if len(name) > 1 and name.isalpha() and set(name) <= declared:
            substitutions[symbol] = s.Mul(*(s.Symbol(c) for c in name))
    if 'e' in declared:
        substitutions[s.E] = s.Symbol('e')
    return value.xreplace(substitutions)


def assignment_rhs(text):
    # A label is not a numeric equality: preserve 2+2=5 for consistency checking.
    sides = split_top(text, '=')
    if len(sides) >= 2 and re.fullmatch(
            r'(?:(?:\\[A-Za-z]+|[A-Za-z][A-Za-z_ ]*)(?:_\{?\w+\}?|\^\{?\d+\}?)?'
            r'(?:\([^()]+\))?|\([A-Za-z, ]+\))', sides[0]):
        return '='.join(sides[1:]).strip()
    return text


def numeric_value(text, task=''):
    """Read a numeric quantity with optional descriptive labels in task units."""
    import sympy as s

    text = assignment_rhs(clean(text))
    text = re.sub(r'\\text\{([^{}]*)\}', r' \1 ', text)
    # Remove thousands separators, not tuple separators (called per component).
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
    label = re.search(r'(?:\s+|/)([A-Za-z][A-Za-z /-]*)\s*$', text)
    if label:
        words = re.findall(r'[A-Za-z]+', label.group(1).lower())
        task_words = set(re.findall(r'[A-Za-z]+', task.lower()))
        aliases = {'hour': {'hour', 'hours', 'hourly'}, 'dollar': {'dollar', 'dollars'},
                   'minute': {'minute', 'minutes'}, 'degree': {'degree', 'degrees'}}
        short_units = {'g': {'gram', 'grams'}, 'h': {'hour', 'hours', 'hourly'},
                       'm': {'meter', 'meters', 'metre', 'metres'}, 's': {'second', 'seconds'}}
        def mentioned(word):
            if word in {'not', 'or', 'except', 'approximately', 'unknown'}:
                return False
            # Never silently erase an algebraic variable as though it were a unit.
            if len(word) == 1:
                return bool(task_words & short_units.get(word, set()))
            if word in task_words:
                return True
            if len(word) > 3 and word.endswith('s') and word[:-1] in task_words:
                return True
            return any(word in forms and task_words & forms for forms in aliases.values())
        if not words or not all(mentioned(w) for w in words):
            raise UnsupportedAnswer('Unrecognized quantity label or units')
        text = text[:label.start()].strip()
    if r'\%' in text:
        if 'percentage point' not in task.lower():
            raise UnsupportedAnswer('Percent versus fraction needs an explicit unit convention')
        text = text.replace(r'\%', '')
    sides = split_top(text, '=')
    values = [parse_math(side, task) for side in sides]
    if any(not isinstance(v, s.Expr) or v.free_symbols or v.is_number is not True for v in values):
        raise UnsupportedAnswer('Expected a complete numeric expression')
    if any(s.simplify(v - values[0]) != 0 for v in values[1:]):
        raise UnsupportedAnswer('Inconsistent numeric equality')
    return values[0]


def tuple_parts(text):
    text = clean(text)
    if text.startswith('(') and text.endswith(')'):
        inside = text[1:-1]
        if len(split_top(inside)) > 1:
            return split_top(inside)
    return None


def tuple_labels_agree(parts, task):
    """Do not erase explicitly reversed labels in a requested ordered tuple."""
    matches = re.findall(r'(?:return|find|counts)\s*\(([^()]+)\)', task, re.IGNORECASE)
    if not matches:
        return True
    labels = [x.strip().lower() for x in matches[-1].split(',')]
    if len(labels) != len(parts) or not all(re.fullmatch(r'[a-z][a-z ]*', x) for x in labels):
        return True
    for i, part in enumerate(parts):
        for j, label in enumerate(labels):
            if i != j and re.search(r'\b' + re.escape(label) + r'\b', part.lower()):
                return False
    return True


def equal_values(left, right):
    import sympy as s
    from math_verify.grader import sympy_expr_eq

    if isinstance(left, tuple) or isinstance(right, tuple):
        return (isinstance(left, tuple) and isinstance(right, tuple)
                and len(left) == len(right)
                and all(equal_values(a, b) for a, b in zip(left, right)))
    if isinstance(left, s.Expr) and isinstance(right, s.Expr):
        if s.simplify(left - right) == 0:
            return True
    return bool(sympy_expr_eq(right, left, float_rounding=6, numeric_precision=15, strict=True))


def compare_answer(answer, reference, task=''):
    import sympy as s

    answer, reference = clean(answer), clean(reference)
    parts = tuple_parts(reference)
    if parts and 'interval' not in task.lower():
        expected = tuple(parse_math(x, task) for x in parts)
        candidate = assignment_rhs(answer)
        if len(parts) == 3 and re.search(r'\\(?:mathbf|hat)\{[ijk]\}', candidate):
            vector = re.sub(r'\\(?:mathbf|hat)\{([ijk])\}', r'v_\1', candidate)
            expr = parse_math(vector, task)
            basis = [s.Symbol('v_' + c) for c in 'ijk']
            values = tuple(s.expand(expr).coeff(v) for v in basis)
            if s.simplify(expr - sum(v*c for v, c in zip(basis, values))) != 0:
                raise UnsupportedAnswer('Unsupported vector representation')
        else:
            candidate = re.sub(r'\\text\{([^{}]*)\}', r' \1 ', candidate)
            raw = tuple_parts(candidate) or split_top(candidate)
            if len(raw) != len(parts):
                return False
            if not tuple_labels_agree(raw, task):
                return False
            if all(isinstance(v, s.Expr) and v.is_number for v in expected):
                values = tuple(numeric_value(x, task) for x in raw)
            else:
                values = tuple(parse_math(assignment_rhs(x), task) for x in raw)
        return equal_values(values, expected)

    if 'interval' in task.lower() and re.fullmatch(r'[\[(].+,.+[\])]', reference):
        bounds = split_top(reference[1:-1])
        expected = s.Interval(*(parse_math(x, task) for x in bounds),
                              left_open=reference[0] == '(', right_open=reference[-1] == ')')
        if re.fullmatch(r'[\[(].+,.+[\])]', answer):
            actual_bounds = split_top(answer[1:-1])
            actual = s.Interval(*(parse_math(x, task) for x in actual_bounds),
                                left_open=answer[0] == '(', right_open=answer[-1] == ')')
        else:
            actual = parse_math(answer, task).as_set()
        return bool(actual == expected)

    expected = parse_math(reference, task)
    if isinstance(expected, s.Equality) and not expected.free_symbols:
        expected = numeric_value(reference, task)
    if isinstance(expected, s.Expr) and expected.is_number and not expected.free_symbols:
        return equal_values(numeric_value(answer, task), expected)
    actual = parse_math(answer, task)
    if isinstance(actual, s.Equality) and not isinstance(expected, s.Equality):
        if isinstance(actual.lhs, s.Symbol):
            actual = actual.rhs
        else:
            raise UnsupportedAnswer('Cannot discard a mathematical equality')
    return equal_values(actual, expected)


def equivalent(item):
    """Process-pool entry point: unsupported/ambiguous output stays unknown."""
    answer, references, task = item
    if not answer:
        return None
    values = []
    for reference in references:
        try:
            values.append(compare_answer(answer, reference, task))
        except Exception:
            values.append(None)
    return True if True in values else None if None in values else False


def form_rule(task):
    task = task.lower()
    if 'into linear factors' in task:
        return 'linear_factors'
    if 'product of two monic quadratic' in task:
        return 'two_monic_quadratics'
    return None


def requested_form(item):
    """Check only explicitly supported output forms; no claim about reasoning."""
    import sympy as s

    answer, task = item
    rule = form_rule(task)
    if not rule or not answer:
        return None
    try:
        expr = parse_math(answer, task)
        if isinstance(expr, s.Equality):
            expr = expr.rhs
        factors = []
        for factor in s.Mul.make_args(expr):
            if factor.is_number:
                continue
            base, power = factor.as_base_exp()
            if not power.is_Integer or not 0 < power <= 20:
                return False
            factors.extend([s.Poly(base, *sorted(expr.free_symbols, key=str))] * int(power))
        if rule == 'linear_factors':
            return bool(factors) and all(p.total_degree() == 1 for p in factors)
        return len(factors) == 2 and all(p.total_degree() == 2 and p.LC() == 1 for p in factors)
    except Exception:
        return None
