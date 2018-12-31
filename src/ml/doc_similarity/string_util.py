import re


def underscore_format(name):
    s = re.sub(r'(\s+|-+|_{2,})', '_', name)
    s = re.sub(r'([a-z])([A-Z])', r'\1_\2', s)
    s = re.sub(r'([A-Z]{2,})([a-z])', r'\1_\2', s)
    return s.lower()
