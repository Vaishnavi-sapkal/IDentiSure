# SIDTD Dataset Analysis

## Fake Annotation Summary

- Total fake annotation files: **1222**
- Records with `second_src != None`: **145**
- Records with `second_field != None`: **145**

## Forgery Types (`ctype`)

| Forgery Type | Count |
|---|---:|
| `Inpaint_and_Rewrite` | 1077 |
| `Crop_and_Replace` | 145 |

## Manipulated Fields

| Field | Count |
|---|---:|
| `gender` | 113 |
| `surname` | 111 |
| `name` | 111 |
| `number` | 104 |
| `expiry_date` | 87 |
| `nationality` | 82 |
| `birth_date` | 66 |
| `issue_date` | 60 |
| `id_number` | 57 |
| `birth_place` | 56 |
| `mrz_line0` | 35 |
| `authority` | 29 |
| `mrz_line1` | 25 |
| `patronymic` | 24 |
| `code` | 23 |
| `type` | 21 |
| `height` | 19 |
| `birth_place_line0` | 19 |
| `birth_date_22` | 17 |
| `surname_second` | 13 |
| `name_eng` | 12 |
| `surname_eng` | 10 |
| `number2` | 10 |
| `name_code` | 9 |
| `number3` | 9 |
| `authority_line1` | 9 |
| `residence_line0` | 9 |
| `birth_date_2` | 8 |
| `issue_place` | 8 |
| `photo` | 7 |
| `signature` | 7 |
| `birth_place_eng` | 7 |
| `birth_place_line1` | 7 |
| `authority_eng` | 6 |
| `expiry_date2` | 6 |
| `nationality/nationality_eng` | 5 |
| `birth_country` | 5 |
| `residence_line1` | 5 |
| `expiry_date_2` | 4 |
| `authority_line0` | 4 |
| `birth_place_line2` | 3 |

## Second Source Usage

145 fake annotations contain a `second_src` value.

Examples:

- `alb_id_23_fake_6_105.json` → `alb_id_87.jpg`
- `alb_id_23_fake_6_106.json` → `alb_id_23.jpg`
- `alb_id_46_fake_6_109.json` → `alb_id_98.jpg`
- `alb_id_46_fake_6_110.json` → `alb_id_46.jpg`
- `alb_id_49_fake_6_114.json` → `alb_id_63.jpg`

## Second Field Usage

145 fake annotations contain a `second_field` value.

Examples:

- `alb_id_23_fake_6_105.json` → `name`
- `alb_id_23_fake_6_106.json` → `name`
- `alb_id_46_fake_6_109.json` → `number`
- `alb_id_46_fake_6_110.json` → `number`
- `alb_id_49_fake_6_114.json` → `nationality`
