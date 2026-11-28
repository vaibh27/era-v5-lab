#!/usr/bin/env python3
"""sa_paradigms.py - Sanskrit marker-word candidates from grammar tables (#675). Drafted with AI, reviewed by a reader.

The forms come from standard paradigms, not from any corpus:
  - pronoun declensions (tad, yad, etad, kim, sarva, anya, idam, asmad, yushmad), all cases and numbers
  - verbs: present, imperfect, optative, imperative, future of common roots (parasmaipada); present and imperfect
    of common atmanepada stems; present of common passive stems; hand tables for athematic roots
  - a few perfect 3rd-person forms, absolutives, infinitives, indeclinables
Every word-final म् also gets its anusvara spelling (तम् -> तं), the common web spelling.

The corpus is used only afterwards, as a safety filter (sa_marker_build.py): a form that occurs in Hindi or Marathi
text is dropped.

Run: python3 sa_paradigms.py  -> prints the number of forms per paradigm group
"""

# ------------------------------------------------------------------ pronouns
# Pronominal declension with stem T: (case, number) order nom, acc, ins, dat, abl, gen, loc x sg, du, pl.
# Forms are written as T + suffix; the nominative singulars are irregular and given per pronoun.
PRON_M = ["{Sm}", "ौ", "े", "म्", "ौ", "ान्", "ेन", "ाभ्याम्", "ैः", "स्मै", "ाभ्याम्", "ेभ्यः",
          "स्मात्", "ाभ्याम्", "ेभ्यः", "स्य", "योः", "ेषाम्", "स्मिन्", "योः", "ेषु"]
PRON_F = ["{Sf}", "े", "ाः", "ाम्", "े", "ाः", "या", "ाभ्याम्", "ाभिः", "स्यै", "ाभ्याम्", "ाभ्यः",
          "स्याः", "ाभ्याम्", "ाभ्यः", "स्याः", "योः", "ासाम्", "स्याम्", "योः", "ासु"]
PRON_N_NOM = ["{Sn}", "े", "ानि"]  # nom = acc; the other cases are as the masculine

PRONOUNS = {  # name: (stem T, nom sg m, nom sg f, nom-acc sg n)
    "tad": ("त", "सः", "सा", "तत्"),
    "yad": ("य", "यः", "या", "यत्"),
    "etad": ("एत", "एषः", "एषा", "एतत्"),
    "kim": ("क", "कः", "का", "किम्"),
    "sarva": ("सर्व", "सर्वः", "सर्वा", "सर्वम्"),
    "anya": ("अन्य", "अन्यः", "अन्या", "अन्यत्"),
}

IRREGULAR_PRONOUNS = {
    "idam m": "अयम् इमौ इमे इमम् इमौ इमान् अनेन आभ्याम् एभिः अस्मै आभ्याम् एभ्यः अस्मात् आभ्याम् एभ्यः अस्य अनयोः एषाम् अस्मिन् अनयोः एषु",
    "idam f": "इयम् इमे इमाः इमाम् इमे इमाः अनया आभ्याम् आभिः अस्यै आभ्याम् आभ्यः अस्याः आभ्याम् आभ्यः अस्याः अनयोः आसाम् अस्याम् अनयोः आसु",
    "idam n": "इदम् इमे इमानि",
    "asmad": "अहम् आवाम् वयम् माम् मा आवाम् नौ अस्मान् नः मया आवाभ्याम् अस्माभिः मह्यम् मे अस्मभ्यम् मत् अस्मत् मम आवयोः अस्माकम् मयि अस्मासु",
    "yushmad": "त्वम् युवाम् यूयम् त्वाम् त्वा युष्मान् वः त्वया युवाभ्याम् युष्माभिः तुभ्यम् ते युष्मभ्यम् त्वत् युष्मत् तव युवयोः युष्माकम् त्वयि युष्मासु",
}

# ------------------------------------------------------------------ verbs
P_PRESENT = ["ति", "तः", "न्ति", "सि", "थः", "थ", "ामि", "ावः", "ामः"]          # stem in -a: भव + ति
P_IMPERFECT = ["त्", "ताम्", "न्", "ः", "तम्", "त", "म्", "ाव", "ाम"]            # augmented stem: अभव + त्
P_OPTATIVE = ["ेत्", "ेताम्", "ेयुः", "ेः", "ेतम्", "ेत", "ेयम्", "ेव", "ेम"]
P_IMPERATIVE = ["तु", "ताम्", "न्तु", "", "तम्", "त", "ानि", "ाव", "ाम"]
A_PRESENT = ["ते", "ेते", "न्ते", "से", "ेथे", "ध्वे", "े", "ावहे", "ामहे"]
A_IMPERFECT = ["त", "ेताम्", "न्त", "थाः", "ेथाम्", "ध्वम्", "े", "ावहि", "ामहि"]

THEMATIC_P = {  # root: (present stem, imperfect stem, future stem)
    "bhU be": ("भव", "अभव", "भविष्य"),
    "gam go": ("गच्छ", "अगच्छ", "गमिष्य"),
    "dRz see": ("पश्य", "अपश्य", "द्रक्ष्य"),
    "vad speak": ("वद", "अवद", "वदिष्य"),
    "sthA stand": ("तिष्ठ", "अतिष्ठ", "स्थास्य"),
    "paTh read": ("पठ", "अपठ", "पठिष्य"),
    "likh write": ("लिख", "अलिख", "लेखिष्य"),
    "ji win": ("जय", "अजय", "जेष्य"),
    "nI lead": ("नय", "अनय", "नेष्य"),
    "iS wish": ("इच्छ", "ऐच्छ", "एषिष्य"),
    "pracch ask": ("पृच्छ", "अपृच्छ", "प्रक्ष्य"),
    "jIv live": ("जीव", "अजीव", "जीविष्य"),
    "vas dwell": ("वस", "अवस", "वत्स्य"),
    "pA drink": ("पिब", "अपिब", "पास्य"),
    "pat fall": ("पत", "अपत", "पतिष्य"),
    "car move": ("चर", "अचर", "चरिष्य"),
    "rakS protect": ("रक्ष", "अरक्ष", "रक्षिष्य"),
    "nam bow": ("नम", "अनम", "नंस्य"),
    "cint think (10)": ("चिन्तय", "अचिन्तय", "चिन्तयिष्य"),
    "kath tell (10)": ("कथय", "अकथय", "कथयिष्य"),
}

THEMATIC_A = {  # atmanepada: (present stem, imperfect stem)
    "labh obtain": ("लभ", "अलभ"),
    "vRt be, exist": ("वर्त", "अवर्त"),
    "jan be born": ("जाय", "अजाय"),
    "man think": ("मन्य", "अमन्य"),
    "vRdh grow": ("वर्ध", "अवर्ध"),
    "bhAS speak": ("भाष", "अभाष"),
    "sev serve": ("सेव", "असेव"),
}

PASSIVE = {  # passive present stems, atmanepada present endings
    "kR done": "क्रिय", "vac said": "उच्य", "dRz seen": "दृश्य", "zru heard": "श्रूय", "jJA known": "ज्ञाय",
    "dA given": "दीय", "kath told": "कथ्य", "paTh read": "पठ्य", "likh written": "लिख्य", "gam gone": "गम्य",
}

ATHEMATIC = {  # hand tables: present, imperfect, optative, imperative (9 forms each), then future stem
    "as be": ("अस्ति स्तः सन्ति असि स्थः स्थ अस्मि स्वः स्मः",
              "आसीत् आस्ताम् आसन् आसीः आस्तम् आस्त आसम् आस्व आस्म",
              "स्यात् स्याताम् स्युः स्याः स्यातम् स्यात स्याम् स्याव स्याम",
              "अस्तु स्ताम् सन्तु एधि स्तम् स्त असानि असाव असाम", ""),
    "kR do": ("करोति कुरुतः कुर्वन्ति करोषि कुरुथः कुरुथ करोमि कुर्वः कुर्मः",
              "अकरोत् अकुरुताम् अकुर्वन् अकरोः अकुरुतम् अकुरुत अकरवम् अकुर्व अकुर्म",
              "कुर्यात् कुर्याताम् कुर्युः कुर्याः कुर्यातम् कुर्यात कुर्याम् कुर्याव कुर्याम",
              "करोतु कुरुताम् कुर्वन्तु कुरु कुरुतम् कुरुत करवाणि करवाव करवाम", "करिष्य"),
    "kR do (A)": ("कुरुते कुर्वाते कुर्वते", "", "", "", ""),
    "dA give": ("ददाति दत्तः ददति ददासि दत्थः दत्थ ददामि दद्वः दद्मः",
                "अददात् अदत्ताम् अददुः अददाः अदत्तम् अदत्त अददाम् अदद्व अदद्म",
                "दद्यात् दद्याताम् दद्युः दद्याः दद्यातम् दद्यात दद्याम् दद्याव दद्याम",
                "ददातु दत्ताम् ददतु देहि दत्तम् दत्त ददानि ददाव ददाम", "दास्य"),
    "jJA know": ("जानाति जानीतः जानन्ति जानासि जानीथः जानीथ जानामि जानीवः जानीमः",
                 "अजानात् अजानीताम् अजानन् अजानाः अजानीतम् अजानीत अजानाम् अजानीव अजानीम",
                 "जानीयात् जानीयाताम् जानीयुः जानीयाः जानीयातम् जानीयात जानीयाम् जानीयाव जानीयाम",
                 "जानातु जानीताम् जानन्तु जानीहि जानीतम् जानीत जानानि जानाव जानाम", "ज्ञास्य"),
    "zru hear": ("शृणोति शृणुतः शृण्वन्ति शृणोषि शृणुथः शृणुथ शृणोमि शृणुवः शृणुमः",
                 "अशृणोत् अशृणुताम् अशृण्वन् अशृणोः अशृणुतम् अशृणुत अशृणवम् अशृणुव अशृणुम",
                 "शृणुयात् शृणुयाताम् शृणुयुः शृणुयाः शृणुयातम् शृणुयात शृणुयाम् शृणुयाव शृणुयाम",
                 "शृणोतु शृणुताम् शृण्वन्तु शृणु शृणुतम् शृणुत शृणवानि शृणवाव शृणवाम", "श्रोष्य"),
}

PERFECT = "उवाच ऊचुः बभूव बभूवुः जगाम जग्मुः ददर्श चकार चक्रुः आस आसुः ददौ ददुः तस्थौ उवास"
ABSOLUTIVE = ("कृत्वा गत्वा दृष्ट्वा श्रुत्वा भूत्वा दत्त्वा ज्ञात्वा स्थित्वा उक्त्वा पठित्वा लिखित्वा जित्वा नीत्वा पीत्वा "
              "आगत्य प्राप्य विहाय आदाय उपगम्य")
INFINITIVE = "कर्तुम् गन्तुम् द्रष्टुम् वक्तुम् पठितुम् ज्ञातुम् दातुम् श्रोतुम् स्थातुम् भवितुम्"
INDECLINABLE = ("तदा यदा कदा सर्वदा ततः यतः कुतः अतः इतः तत्र यत्र कुत्र अत्र सर्वत्र अन्यत्र इह अद्य श्वः ह्यः "
                "इदानीम् अधुना पुनः एव अपि इति च वा तु हि खलु किल ननु नूनम् इव यथा तथा कथम् यावत् तावत् चेत् "
                "यदि न मा सह विना प्रति उपरि अधः अन्तः बहिः सम्यक् शनैः सदा नित्यम् सहसा वै उत अथ अथवा "
                "नास्ति नैव चापि चैव तथापि यद्यपि तर्हि")


def build(dedup=True):
    """[(form, group, source)] in a fixed order; with dedup, each form once (first group wins)."""
    out = []

    def add(group, source, forms):
        for f in forms:
            if f:
                out.append((f, group, source))

    for name, (t, sm, sf, sn) in PRONOUNS.items():
        add("pronoun", f"{name} m", [s.format(Sm=sm) if "{" in s else t + s for s in PRON_M])
        add("pronoun", f"{name} f", [s.format(Sf=sf) if "{" in s else t + s for s in PRON_F])
        add("pronoun", f"{name} n", [s.format(Sn=sn) if "{" in s else t + s for s in PRON_N_NOM])
    for name, forms in IRREGULAR_PRONOUNS.items():
        add("pronoun", name, forms.split())
    for name, (pres, impf, fut) in THEMATIC_P.items():
        add("verb", f"{name} present", [pres + e for e in P_PRESENT])
        add("verb", f"{name} imperfect", [impf + e for e in P_IMPERFECT])
        add("verb", f"{name} optative", [pres + e for e in P_OPTATIVE])
        add("verb", f"{name} imperative", [pres + e for e in P_IMPERATIVE])
        add("verb", f"{name} future", [fut + e for e in P_PRESENT])
    for name, (pres, impf) in THEMATIC_A.items():
        add("verb", f"{name} present (A)", [pres + e for e in A_PRESENT])
        add("verb", f"{name} imperfect (A)", [impf + e for e in A_IMPERFECT])
    for name, stem in PASSIVE.items():
        add("verb", f"{name} passive", [stem + e for e in A_PRESENT])
    for name, (pres, impf, opt, imv, fut) in ATHEMATIC.items():
        for tense, forms in (("present", pres), ("imperfect", impf), ("optative", opt), ("imperative", imv)):
            add("verb", f"{name} {tense}", forms.split())
        if fut:
            add("verb", f"{name} future", [fut + e for e in P_PRESENT])
    add("verb", "perfect", PERFECT.split())
    add("other", "absolutive", ABSOLUTIVE.split())
    add("other", "infinitive", INFINITIVE.split())
    add("other", "indeclinable", INDECLINABLE.split())

    # anusvara spelling of a final m: तम् -> तं
    out += [(f[:-2] + "ं", g, s + " (anusvara)") for f, g, s in list(out) if f.endswith("म्")]
    if not dedup:
        return out
    seen, uniq = set(), []
    for row in out:
        if row[0] not in seen:
            seen.add(row[0])
            uniq.append(row)
    return uniq


if __name__ == "__main__":
    from collections import Counter
    rows = build()
    print(len(rows), "forms:", dict(Counter(g for _, g, _ in rows)))
