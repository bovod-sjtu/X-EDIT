"""Canonical accent/dialect labels used by the voice reference library."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


WU_CHINESE_FAMILY = "Wu Chinese"
WU_SUBDIALECTS = {
    "Shanghai Dialect",
    "Suzhou Dialect",
    "Ningbo Dialect",
}


@dataclass(frozen=True)
class AccentDefinition:
    canonical_name: str
    label_type: str
    language: str
    locale: str | None = None
    region: str | None = None
    family: str | None = None
    aliases: tuple[str, ...] = ()


ACCENT_DEFINITIONS: tuple[AccentDefinition, ...] = (
    AccentDefinition(
        "General American English",
        "accent",
        "en",
        "en-US",
        "United States",
        "English",
        (
            "american",
            "american accent",
            "american english",
            "general american",
            "standard english",
            "us english",
            "u.s. english",
        ),
    ),
    AccentDefinition(
        "Southern American English",
        "accent",
        "en",
        "en-US",
        "Southern United States",
        "English",
        ("southern american", "southern american accent", "southern accent"),
    ),
    AccentDefinition(
        "British English",
        "accent",
        "en",
        "en-GB",
        "United Kingdom",
        "English",
        ("british", "british accent", "british english", "received pronunciation", "rp accent"),
    ),
    AccentDefinition("Australian English", "accent", "en", "en-AU", "Australia", "English", ("australian", "australian accent")),
    AccentDefinition("Indian English", "accent", "en", "en-IN", "India", "English", ("indian", "indian accent")),
    AccentDefinition("French-accented English", "accent", "en", None, "France", "English", ("french accent", "french-accented english")),
    AccentDefinition("Chinese-accented English", "accent", "en", None, "China", "English", ("chinese accent", "mandarin accented english")),
    AccentDefinition("German-accented English", "accent", "en", None, "Germany", "English", ("german accent", "german-accented english")),
    AccentDefinition("Spanish-accented English", "accent", "en", None, "Spain/Latin America", "English", ("spanish accent", "spanish-accented english")),
    AccentDefinition("Russian-accented English", "accent", "en", None, "Russia", "English", ("russian accent", "russian-accented english")),
    AccentDefinition("Ukrainian-accented English", "accent", "en", None, "Ukraine", "English", ("ukrainian accent", "ukrainian-accented english")),
    AccentDefinition("Dari-accented English", "accent", "en", None, "Afghanistan", "English", ("dari accent", "dari-accented english")),
    AccentDefinition("Japanese-accented English", "accent", "en", None, "Japan", "English", ("japanese accent", "japanese-accented english")),
    AccentDefinition("Korean-accented English", "accent", "en", None, "Korea", "English", ("korean accent", "korean-accented english")),
    AccentDefinition(
        "Thai-accented English",
        "accent",
        "en",
        None,
        "Thailand",
        "English",
        ("thai", "thailand", "thai english", "thai accent", "thai-accented english", "thai style"),
    ),
    AccentDefinition("Scottish English", "accent", "en", "en-GB", "Scotland", "English", ("scottish", "scottish accent")),
    AccentDefinition("Irish English", "accent", "en", "en-IE", "Ireland", "English", ("irish", "irish accent")),
    AccentDefinition("Northern Irish English", "accent", "en", "en-GB", "Northern Ireland", "English", ("northern irish", "northern irish accent", "northernirish")),
    AccentDefinition("Welsh English", "accent", "en", "en-GB", "Wales", "English", ("welsh", "welsh accent")),
    AccentDefinition("Canadian English", "accent", "en", "en-CA", "Canada", "English", ("canadian", "canadian accent")),
    AccentDefinition("New Zealand English", "accent", "en", "en-NZ", "New Zealand", "English", ("new zealand", "new zealand accent")),
    AccentDefinition("South African English", "accent", "en", "en-ZA", "South Africa", "English", ("south african", "south african accent")),
    AccentDefinition(
        "Native Korean",
        "language",
        "ko",
        "ko-KR",
        "Korea",
        "Korean",
        ("native korean", "standard korean", "korean language", "korean speech", "韩语", "韩国语", "标准韩语"),
    ),
    AccentDefinition(
        "Kansai Japanese",
        "dialect",
        "ja",
        "ja-JP",
        "Kansai",
        "Japanese",
        ("kansai japanese", "kansai dialect", "kansai accent", "関西弁", "关西腔", "关西方言"),
    ),
    AccentDefinition(
        "Standard Mandarin",
        "dialect",
        "zh",
        "zh-CN",
        "China",
        "Mandarin",
        ("standard mandarin", "mandarin", "mandarin chinese", "putonghua", "普通话", "标准普通话"),
    ),
    AccentDefinition(
        "Taiwanese Mandarin",
        "accent",
        "zh",
        "zh-TW",
        "Taiwan",
        "Mandarin",
        ("taiwanese mandarin", "taiwan accent", "taiwanese accent", "台湾腔", "台湾普通话", "台灣國語"),
    ),
    AccentDefinition(
        "Cantonese",
        "dialect",
        "zh",
        "yue-CN",
        "Guangdong/Hong Kong/Macau",
        "Yue Chinese",
        ("cantonese", "cantonese chinese", "yue chinese", "粤语", "广东话", "廣東話"),
    ),
    AccentDefinition(
        "Dongbei Dialect",
        "dialect",
        "zh",
        "zh-CN",
        "Northeast China",
        "Northeastern Mandarin",
        ("dongbei", "dongbei dialect", "northeastern mandarin", "东北话", "东北方言", "东北官话"),
    ),
    AccentDefinition(
        WU_CHINESE_FAMILY,
        "dialect",
        "zh",
        "zh-CN",
        "Jiangsu/Zhejiang/Shanghai",
        "Wu Chinese",
        ("wu chinese", "wu dialect", "wuyu", "wu yu", "吴语", "吴方言"),
    ),
    AccentDefinition(
        "Shanghai Dialect",
        "dialect",
        "zh",
        "wuu-CN",
        "Shanghai",
        WU_CHINESE_FAMILY,
        ("shanghai dialect", "shanghainese", "shanghai wu", "上海话", "上海方言"),
    ),
    AccentDefinition(
        "Suzhou Dialect",
        "dialect",
        "zh",
        "wuu-CN",
        "Suzhou",
        WU_CHINESE_FAMILY,
        ("suzhou dialect", "suzhou accent", "suzhou wu", "suzhou wuyu", "苏州话", "苏州方言"),
    ),
    AccentDefinition(
        "Ningbo Dialect",
        "dialect",
        "zh",
        "wuu-CN",
        "Ningbo",
        WU_CHINESE_FAMILY,
        ("ningbo dialect", "ningbo accent", "ningbo wu", "ningbo wuyu", "宁波话", "宁波方言"),
    ),
    AccentDefinition(
        "Minnan Dialect",
        "dialect",
        "zh",
        "nan-CN",
        "Southern Fujian/Taiwan",
        "Southern Min",
        ("minnan", "minnan dialect", "southern min", "hokkien", "闽南语", "闽南话", "闽南方言"),
    ),
    AccentDefinition(
        "Sichuan Dialect",
        "dialect",
        "zh",
        "zh-CN",
        "Sichuan/Chongqing",
        "Southwestern Mandarin",
        ("sichuan", "sichuan dialect", "sichuanese", "四川话", "四川方言", "西南官话"),
    ),
    AccentDefinition(
        "Beijing Dialect",
        "dialect",
        "zh",
        "zh-CN",
        "Beijing",
        "Mandarin",
        ("beijing dialect", "beijing accent", "北京话", "北京方言"),
    ),
    AccentDefinition(
        "Beijing Guanhua",
        "dialect",
        "zh",
        "zh-CN",
        "Beijing",
        "Mandarin",
        ("beijing guanhua", "beijing mandarin", "北京官话", "北京官話"),
    ),
)


COMPATIBLE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"General American English", "Southern American English", "Canadian English"}),
    frozenset({"British English", "Scottish English", "Irish English", "Northern Irish English", "Welsh English", "South African English"}),
    frozenset({"Australian English", "New Zealand English"}),
    frozenset({"Chinese-accented English", "Standard Mandarin"}),
    frozenset({"Korean-accented English", "Native Korean"}),
    frozenset({"Japanese-accented English", "Kansai Japanese"}),
    frozenset({"Thai-accented English"}),
    frozenset({"Dongbei Dialect", "Standard Mandarin"}),
    frozenset({"Sichuan Dialect", "Standard Mandarin"}),
    frozenset({"Taiwanese Mandarin", "Standard Mandarin", "Chinese-accented English"}),
    frozenset({"Beijing Dialect", "Beijing Guanhua", "Standard Mandarin"}),
    frozenset({WU_CHINESE_FAMILY, *WU_SUBDIALECTS}),
)


GENERIC_LABELS = {"", "unknown", "none", "other", "n/a", "native", "english", "native english"}


def label_key(value: str) -> str:
    value = value.lower().replace("u.s.", "us").replace("u.k.", "uk")
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


_ALIAS_TO_CANONICAL = {
    label_key(alias): definition.canonical_name
    for definition in ACCENT_DEFINITIONS
    for alias in (definition.canonical_name, *definition.aliases)
}
_SORTED_ALIAS_KEYS = sorted(_ALIAS_TO_CANONICAL, key=len, reverse=True)


def list_accent_definitions() -> list[AccentDefinition]:
    return list(ACCENT_DEFINITIONS)


def list_canonical_labels(*, include_family_fallbacks: bool = True) -> list[str]:
    labels = [definition.canonical_name for definition in ACCENT_DEFINITIONS]
    if include_family_fallbacks:
        return labels
    return [label for label in labels if label != WU_CHINESE_FAMILY]


def accent_definition(value: str | None) -> AccentDefinition | None:
    canonical = canonical_label(value)
    if not canonical:
        return None
    for definition in ACCENT_DEFINITIONS:
        if definition.canonical_name == canonical:
            return definition
    return None


def canonical_label(value: object) -> str | None:
    key = label_key(str(value or ""))
    if not key or key in GENERIC_LABELS:
        return None
    if key in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[key]
    for alias_key in _SORTED_ALIAS_KEYS:
        if _contains_alias(key, alias_key):
            return _ALIAS_TO_CANONICAL[alias_key]
    return None


def extract_mentions(text: str) -> list[str]:
    key = label_key(text)
    if not key:
        return []
    mentions: list[str] = []
    for alias_key in _SORTED_ALIAS_KEYS:
        if _contains_alias(key, alias_key):
            canonical = _ALIAS_TO_CANONICAL[alias_key]
            if canonical not in mentions:
                mentions.append(canonical)
    return mentions


def _contains_alias(text_key: str, alias_key: str) -> bool:
    if re.search(r"[\u4e00-\u9fff]", alias_key):
        return alias_key in text_key
    return f" {alias_key} " in f" {text_key} "


def accents_compatible(left: str | None, right: str | None) -> bool:
    left_norm = canonical_label(left) or left
    right_norm = canonical_label(right) or right
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    return any(left_norm in group and right_norm in group for group in COMPATIBLE_GROUPS)


def is_wu_subdialect(value: str | None) -> bool:
    return (canonical_label(value) or value) in WU_SUBDIALECTS


def requires_fine_label(value: str | None) -> bool:
    return (canonical_label(value) or value) == WU_CHINESE_FAMILY


def best_label_for_request(text: str) -> str | None:
    target_fragment = _target_label_fragment(text)
    if target_fragment:
        target_mentions = extract_mentions(target_fragment)
        for mention in target_mentions:
            if mention != WU_CHINESE_FAMILY:
                return mention
        if target_mentions:
            return target_mentions[0]
    mentions = extract_mentions(text)
    if not mentions:
        return None
    for mention in mentions:
        if mention != WU_CHINESE_FAMILY:
            return mention
    return mentions[0]


_TARGET_LABEL_DIRECTIVE_RE = re.compile(
    r"(?:->|=>|→|\b(?:to|into)\b|转换成|轉換成|转成|轉成|改成|改为|改為|变成|變成|换成|換成|翻译成|翻譯成)"
)


def _target_label_fragment(text: str) -> str | None:
    matches = list(_TARGET_LABEL_DIRECTIVE_RE.finditer(text))
    if not matches:
        return None
    fragment = text[matches[-1].end() :]
    return fragment.strip(" .。!！?？:：,，;；") or None


def reduce_compatible_labels(labels: Iterable[str]) -> str | None:
    unique: list[str] = []
    for label in labels:
        canonical = canonical_label(label) or label
        if canonical and canonical not in unique:
            unique.append(canonical)
    if not unique:
        return None
    for label in unique:
        if all(accents_compatible(label, other) for other in unique):
            if label == WU_CHINESE_FAMILY:
                fine = [item for item in unique if is_wu_subdialect(item)]
                return fine[0] if fine else label
            return label
    return None
