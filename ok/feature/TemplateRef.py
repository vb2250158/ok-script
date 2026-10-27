from ok.feature.VisualRuleTemplate import MIN_VISUAL_RULE_SAMPLES, visual_rule_name_for_template


TEMPLATE_REF_TYPE_TEMPLATE = "template"
TEMPLATE_REF_TYPE_RULE = "rule"


def template_ref(name, ref_type=TEMPLATE_REF_TYPE_TEMPLATE):
    name = str(name or "").strip()
    ref_type = TEMPLATE_REF_TYPE_RULE if ref_type == TEMPLATE_REF_TYPE_RULE else TEMPLATE_REF_TYPE_TEMPLATE
    return {"name": name, "type": ref_type}


def template_ref_name(value):
    if isinstance(value, dict):
        return str(value.get("name") or "").strip()
    return str(value or "").strip()


def template_ref_type(value, default=TEMPLATE_REF_TYPE_TEMPLATE):
    if isinstance(value, dict):
        ref_type = str(value.get("type") or "").strip()
        if ref_type in {TEMPLATE_REF_TYPE_RULE, TEMPLATE_REF_TYPE_TEMPLATE}:
            return ref_type
    return default


def sanitize_template_ref(value, rule_names=None):
    name = template_ref_name(value)
    if not name:
        return None
    if isinstance(value, dict):
        ref_type = template_ref_type(value, "")
        if ref_type not in {TEMPLATE_REF_TYPE_RULE, TEMPLATE_REF_TYPE_TEMPLATE}:
            rule_names = set(rule_names or [])
            ref_type = TEMPLATE_REF_TYPE_RULE if name in rule_names else TEMPLATE_REF_TYPE_TEMPLATE
    else:
        rule_names = set(rule_names or [])
        ref_type = TEMPLATE_REF_TYPE_RULE if name in rule_names else TEMPLATE_REF_TYPE_TEMPLATE
    return template_ref(name, ref_type)


def sanitize_template_refs(values, rule_names=None):
    refs = []
    seen = set()
    for value in values or []:
        ref = sanitize_template_ref(value, rule_names)
        if not ref:
            continue
        key = template_ref_key(ref)
        if key in seen:
            continue
        seen.add(key)
        refs.append(ref)
    return refs


def template_ref_key(value):
    ref = sanitize_template_ref(value)
    if not ref:
        return ("", "")
    return (ref["type"], ref["name"])


def template_param_value(value):
    ref = sanitize_template_ref(value)
    if not ref:
        return ""
    if ref["type"] == TEMPLATE_REF_TYPE_RULE:
        return ref
    return ref["name"]


def visual_rule_template_names_from_samples(template_names, min_samples=MIN_VISUAL_RULE_SAMPLES):
    group_counts = {}
    for template_name in template_names or []:
        rule_name = visual_rule_name_for_template(template_name)
        if rule_name:
            group_counts[rule_name] = group_counts.get(rule_name, 0) + 1
    return {
        rule_name
        for rule_name, count in group_counts.items()
        if count >= min_samples
    }


def template_choice_refs_from_names(template_names, rule_names=None):
    raw_names = {str(name or "").strip() for name in template_names or [] if str(name or "").strip()}
    rule_names = (
        {str(name or "").strip() for name in rule_names or [] if str(name or "").strip()}
        if rule_names is not None
        else visual_rule_template_names_from_samples(raw_names)
    )
    refs = [
        template_ref(name, TEMPLATE_REF_TYPE_RULE)
        for name in rule_names
    ] + [
        template_ref(name, TEMPLATE_REF_TYPE_TEMPLATE)
        for name in raw_names
    ]
    return sorted(
        sanitize_template_refs(refs),
        key=lambda ref: (str(ref.get("name") or ""), 0 if ref.get("type") == TEMPLATE_REF_TYPE_RULE else 1),
    )
