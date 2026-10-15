#!/usr/bin/env python3
"""Navoiy v1.0.0 PIN fix and no-regression on all 20 telecom inputs."""
import json, pathlib
from telecom_normalizer import TelecomUzbekNormalizer, PIN_SPOKEN
HERE=pathlib.Path(__file__).resolve().parent
assert PIN_SPOKEN=='pi ay en'
n=TelecomUzbekNormalizer(lambda s:s.strip())
cases={
 'PIN-kodingizni hech kimga aytmang.':'pi ay en kodingizni hech kimga aytmang.',
 'pin-kodingizni hech kimga aytmang.':'pi ay en kodingizni hech kimga aytmang.',
 'PIN-kod sozlamasi.':'pi ay en kod sozlamasi.',
 'PIN kodi.':'pi ay en kodi.',
 'PIN maxfiy.':'pi ay en maxfiy.',
 'PIN 1234.':'pi ay en 1234.',
 'spinning orqali':'spinning orqali',
 '90 123 45 67':"to'qson, bir yuz yigirma uch, qirq besh, oltmish yetti",
 'Tasdiqlash kodi 4 8 2 7 1 9':"Tasdiqlash kodi to'rt sakkiz ikki yetti bir to'qqiz",
 'Telefoningizda LTE yoki 4G tarmog\'ini yoqing.':"Telefoningizda el ti i yoki to'rt ji tarmog'ini yoqing.",
}
for inp,want in cases.items():
 got=n.normalize(inp)
 assert got==want,(inp,want,got)
original=json.loads((HERE/'telecom20_suite.json').read_text())['cases']
golden=json.loads((HERE/'telecom20_expected_normalized.json').read_text())
assert len(original)==len(golden)==20
for i,(a,b) in enumerate(zip(original,golden)):
 assert a['label']==b['label'] and a['raw']==b['raw']
 if a['label']=='19_pin_instruction':
  assert n.normalize(a['raw'])==b['normalized']
  assert b['telecom_entities'][0]['kind']=='pin_compound'
 else:
  # Verify the PIN hotfix doesn't change any other original input, *before*
  # uztts ordinary number normalization (performed at runtime).
  assert not any(e['kind'].startswith('pin') for e in n.normalize_with_trace(a['raw'])[1]),a
numeric={r['label']:r['normalized'] for r in golden}
assert "qirq to'qqiz ming to'qqiz yuz" in numeric['04_price']
assert "to'qson, bir yuz yigirma uch, qirq besh, oltmish yetti" in numeric['05_phone_number']
assert "yetmish besh ming" in numeric['12_tariff_combo']
assert "ming besh yuz" in numeric['10_minutes']
assert "to'rt sakkiz ikki yetti bir to'qqiz" in numeric['18_verification_code']
print('PASS: PIN explicit letter spelling, 20 original telecom cases, numeric golden review')
