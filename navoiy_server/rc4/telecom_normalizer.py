#!/usr/bin/env python3
import re
from dataclasses import dataclass
from typing import Callable, List

ONES={0:"nol",1:"bir",2:"ikki",3:"uch",4:"to'rt",5:"besh",6:"olti",7:"yetti",8:"sakkiz",9:"to'qqiz"}
TENS={10:"o'n",20:"yigirma",30:"o'ttiz",40:"qirq",50:"ellik",60:"oltmish",70:"yetmish",80:"sakson",90:"to'qson"}
TERM={"4g":"to'rt ji","5g":"besh ji","lte":"el ti i","sms":"es em es","esim":"i sim","sim":"sim","otp":"ou ti pi"}

PHONE_INTL_RE=re.compile(r"(?<!\d)(?P<prefix>\+?998)[\s-]+(?P<a>\d{2})[\s-]+(?P<b>\d{3})[\s-]+(?P<c>\d{2})[\s-]+(?P<d>\d{2})(?!\d)")
PHONE_LOCAL_RE=re.compile(r"(?<!\d)(?P<a>\d{2})[\s-]+(?P<b>\d{3})[\s-]+(?P<c>\d{2})[\s-]+(?P<d>\d{2})(?!\d)")
USSD_RE=re.compile(r"(?<!\w)\*(?P<digits>\d{2,4})#(?!\w)")
# Spell the PIN acronym as P-I-N in English letter names, matching LTE/SMS convention.
# Treat common Uzbek compounds as a single protected entity so `-kodingizni`
# does not get attached to the last spelled letter as a single TTS word.
PIN_SPOKEN = "pi ay en"
PIN_COMPOUND_RE=re.compile(r"(?i)\bPIN[\s-]+(?P<code>kod[a-zʻʼ’']*)\b")
PIN_WORD_RE=re.compile(r"(?i)\bPIN\b")
TECH_RE=re.compile(r"(?i)\b(?:4G|5G|LTE|SMS|eSIM|SIM|OTP)\b")
CODE_RE=re.compile(r"(?i)(?P<label>\b(?:tasdiqlash\s+kodi|verification\s+code|otp\s+kodi|kod(?:i)?))(?P<sep>\s*[:\-]?\s*)(?P<digits>\d(?:[\s-]*\d){3,7})")

def cardinal(n:int)->str:
    if n<10:return ONES[n]
    if n<100:
        t=(n//10)*10; o=n%10
        return TENS[t] if o==0 else f"{TENS[t]} {ONES[o]}"
    h=n//100; r=n%100
    return f"{ONES[h]} yuz" if r==0 else f"{ONES[h]} yuz {cardinal(r)}"

def digits(s): return " ".join(ONES[int(ch)] for ch in s if ch.isdigit())
def phone_groups(gs): return ", ".join(cardinal(int(g)) for g in gs)

@dataclass
class Entity:
    start:int; end:int; spoken:str; kind:str; raw:str

class TelecomUzbekNormalizer:
    def __init__(self, base_normalize:Callable[[str],str]): self.base=base_normalize

    def _entities(self,text)->List[Entity]:
        c=[]
        for m in PHONE_INTL_RE.finditer(text):
            p=cardinal(int(m.group("prefix").lstrip("+")))
            if m.group("prefix").startswith("+"): p="plus "+p
            c.append(Entity(m.start(),m.end(),f"{p}, "+phone_groups([m.group("a"),m.group("b"),m.group("c"),m.group("d")]),"phone_international",m.group(0)))
        for m in PHONE_LOCAL_RE.finditer(text):
            c.append(Entity(m.start(),m.end(),phone_groups([m.group("a"),m.group("b"),m.group("c"),m.group("d")]),"phone_local",m.group(0)))
        for m in PIN_COMPOUND_RE.finditer(text):
            c.append(Entity(m.start(),m.end(),f"{PIN_SPOKEN} {m.group('code')}","pin_compound",m.group(0)))
        for m in PIN_WORD_RE.finditer(text):
            c.append(Entity(m.start(),m.end(),PIN_SPOKEN,"pin_acronym",m.group(0)))
        for m in USSD_RE.finditer(text):
            c.append(Entity(m.start(),m.end(),f"yulduzcha {digits(m.group('digits'))} panjara","ussd",m.group(0)))
        for m in CODE_RE.finditer(text):
            c.append(Entity(m.start(),m.end(),f"{m.group('label')} {digits(m.group('digits'))}","code",m.group(0)))
        for m in TECH_RE.finditer(text):
            c.append(Entity(m.start(),m.end(),TERM[m.group(0).lower()],"technology",m.group(0)))
        c.sort(key=lambda e:(e.start,-(e.end-e.start)))
        out=[]; end=-1
        for e in c:
            if e.start>=end:
                out.append(e); end=e.end
        return out

    def _normalize_ordinary_span(self, span):
        """Normalize ordinary text while preserving one boundary space."""
        if not span:
            return ""
        leading_ws = bool(re.match(r"^\s", span))
        trailing_ws = bool(re.search(r"\s$", span))
        core = span.strip()
        norm = self.base(core) if core else ""
        if leading_ws and norm:
            norm = " " + norm
        if trailing_ws and norm:
            norm = norm + " "
        return norm

    def normalize_with_trace(self,text):
        ents=self._entities(text)
        if not ents:
            return self.base(text),[]

        parts=[]; trace=[]; cur=0
        for e in ents:
            if e.start>cur:
                parts.append(self._normalize_ordinary_span(text[cur:e.start]))
            parts.append(e.spoken)
            trace.append({"kind":e.kind,"raw":e.raw,"spoken":e.spoken})
            cur=e.end

        if cur<len(text):
            parts.append(self._normalize_ordinary_span(text[cur:]))

        out="".join(parts)
        out=re.sub(r"[ \t]+"," ",out)
        out=re.sub(r"\s+([,.;:!?])",r"\1",out)
        return out.strip(),trace

    def normalize(self,text): return self.normalize_with_trace(text)[0]

def self_test():
    # Simulate uztts trimming each fragment so boundary bugs are caught here.
    n=TelecomUzbekNormalizer(lambda x:x.strip())
    cases=[
      ("Sizning aloqa raqamingiz 90 123 45 67.","Sizning aloqa raqamingiz to'qson, bir yuz yigirma uch, qirq besh, oltmish yetti."),
      ("Telefoningizda LTE yoki 4G tarmog'ini yoqing.","Telefoningizda el ti i yoki to'rt ji tarmog'ini yoqing."),
      ("Tasdiqlash kodi 482719.","Tasdiqlash kodi to'rt sakkiz ikki yetti bir to'qqiz."),
      ("*102# kodini tering.","yulduzcha bir nol ikki panjara kodini tering."),
      ("Tarif doirasida 500 SMS mavjud.","Tarif doirasida 500 es em es mavjud."),
      ("SIM-karta ishlaydi.","sim-karta ishlaydi."),
      ("PIN-kodingizni aytmang.","pi ay en kodingizni aytmang."),
      ("pin-kodingizni hech kimga aytmang.","pi ay en kodingizni hech kimga aytmang."),
      ("PIN-kod sozlamasi.","pi ay en kod sozlamasi."),
      ("PIN kodi.","pi ay en kodi."),
      ("PIN maxfiy.","pi ay en maxfiy."),
      ("SMS orqali xabar beriladi.","es em es orqali xabar beriladi."),
    ]
    for raw,exp in cases:
        got=n.normalize(raw)
        assert got==exp,(raw,exp,got)
    print(f"PASS: {len(cases)} tests")

if __name__=="__main__": self_test()
