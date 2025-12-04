#!/usr/bin/env python3

"""
================================================================================
MEDICAL PRESCRIPTION ANALYZER - COMPLETE ALL-IN-ONE APPLICATION WITH DOSAGE
================================================================================
DISCLAIMER: Educational purposes only - NOT for clinical use
Always consult healthcare professionals before taking any medication
================================================================================
INSTALLATION INSTRUCTIONS
================================================================================
Step 1: Install Python Dependencies
pip install streamlit pillow pytesseract opencv-python numpy easyocr

Step 2: Install Tesseract OCR (Already installed - good!)
Verify installation:
tesseract --version

If needed:
WINDOWS:
- Download: https://github.com/UB-Mannheim/tesseract/wiki
- Run installer (default path: C:\Program Files\Tesseract-OCR)
MAC:
brew install tesseract
LINUX (Ubuntu/Debian):
sudo apt-get install tesseract-ocr

Step 3: Run the Application
streamlit run prescription_analyzer.py

Step 4: Open in Browser
http://localhost:8501
================================================================================
FEATURES & USAGE
================================================================================
📸 PRESCRIPTION OCR TAB:
1. Upload a handwritten prescription image OR use camera
2. Enter patient age
3. Click "🔍 Analyze Prescription"
4. System extracts ALL drugs from handwriting (EasyOCR + Tesseract fallback)
5. Shows dosages, frequencies, interactions
6. Displays age-appropriate mg/kg dosage recommendations with warnings
7. Shows alternatives if conflicts exist

🔍 INTERACTION CHECKER TAB:
1. Enter multiple drug names (comma/line separated)
2. Click "Check"
3. View all interactions with severity levels

📏 DOSAGE INFO TAB:
1. Select any drug from dropdown
2. Enter patient age
3. View mg/kg dosage recommendations and warnings
4. See drug information and alternatives

🏥 SAFETY CHECK TAB:
1. Select patient's medical conditions
2. Enter medications
3. Click "Check Safety"
4. See contraindications and warnings

================================================================================
END OF DOCUMENTATION - CODE STARTS BELOW
================================================================================
"""

import streamlit as st
import re
from PIL import Image
import numpy as np
import cv2

# ============================================================================ 
# EasyOCR Import
# ============================================================================ 
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    OCR_READER = easyocr.Reader(['en'], gpu=False)
except ImportError:
    EASYOCR_AVAILABLE = False
    OCR_READER = None

# ============================================================================ 
# Tesseract OCR Import
# ============================================================================ 
try:
    import pytesseract
    PYTESSERACT_OCR_AVAILABLE = True
except ImportError:
    PYTESSERACT_OCR_AVAILABLE = False

# ============================================================================ 
# DRUG DATABASE
# ============================================================================ 
DRUGS = {
    "amoxicillin": {"class": "antibiotic", "generic": "amoxicillin"},
    "azithromycin": {"class": "antibiotic", "generic": "azithromycin"},
    "clindamycin": {"class": "antibiotic", "generic": "clindamycin"},
    "ceftriaxone": {"class": "antibiotic", "generic": "ceftriaxone"},
    "amoxicillin-clavulanic acid": {"class": "antibiotic", "generic": "amoxicillin-clavulanic acid"},
    "paracetamol": {"class": "analgesic", "generic": "acetaminophen"},
    "acetaminophen": {"class": "analgesic", "generic": "acetaminophen"},
    "ibuprofen": {"class": "nsaid", "generic": "ibuprofen"},
    "diclofenac": {"class": "nsaid", "generic": "diclofenac"},
    "naproxen": {"class": "nsaid", "generic": "naproxen"},
    "aspirin": {"class": "nsaid", "generic": "aspirin"},
    "morphine": {"class": "opioid", "generic": "morphine"},
    "cetirizine": {"class": "antihistamine", "generic": "cetirizine"},
    "loratadine": {"class": "antihistamine", "generic": "loratadine"},
    "fexofenadine": {"class": "antihistamine", "generic": "fexofenadine"},
    "metformin": {"class": "antidiabetic", "generic": "metformin"},
    "atorvastatin": {"class": "statin", "generic": "atorvastatin"},
    "rosuvastatin": {"class": "statin", "generic": "rosuvastatin"},
    "clopidogrel": {"class": "antiplatelet", "generic": "clopidogrel"},
    "amlodipine": {"class": "calcium_blocker", "generic": "amlodipine"},
    "losartan": {"class": "arb", "generic": "losartan"},
    "telmisartan": {"class": "arb", "generic": "telmisartan"},
    "enalapril": {"class": "ace_inhibitor", "generic": "enalapril"},
    "ramipril": {"class": "ace_inhibitor", "generic": "ramipril"},
    "carvedilol": {"class": "beta_blocker", "generic": "carvedilol"},
    "hydrochlorothiazide": {"class": "diuretic", "generic": "hydrochlorothiazide"},
    "furosemide": {"class": "diuretic", "generic": "furosemide"},
    "spironolactone": {"class": "diuretic", "generic": "spironolactone"},
    "omeprazole": {"class": "ppi", "generic": "omeprazole"},
    "pantoprazole": {"class": "ppi", "generic": "pantoprazole"},
    "ondansetron": {"class": "antiemetic", "generic": "ondansetron"},
    "loperamide": {"class": "antidiarrheal", "generic": "loperamide"},
    "haloperidol": {"class": "antipsychotic", "generic": "haloperidol"},
    "gabapentin": {"class": "anticonvulsant", "generic": "gabapentin"},
    "pregabalin": {"class": "anticonvulsant", "generic": "pregabalin"},
    "donepezil": {"class": "cholinesterase", "generic": "donepezil"},
    "theophylline": {"class": "bronchodilator", "generic": "theophylline"},
    "dextromethorphan": {"class": "antitussive", "generic": "dextromethorphan"},
    "ipratropium": {"class": "anticholinergic", "generic": "ipratropium"},
    "multivitamin": {"class": "supplement", "generic": "multivitamin"},
    "prednisolone": {"class": "corticosteroid", "generic": "prednisolone"},
    "dexamethasone": {"class": "corticosteroid", "generic": "dexamethasone"},
    "hydrocortisone": {"class": "corticosteroid", "generic": "hydrocortisone"},
    "warfarin": {"class": "anticoagulant", "generic": "warfarin"},
}

# ============================================================================ 
# DRUG INTERACTIONS
# ============================================================================ 
INTERACTIONS = {
    ("aspirin", "ibuprofen"): {"severity": "high", "effect": "GI bleeding risk, reduced efficacy"},
    ("aspirin", "diclofenac"): {"severity": "high", "effect": "Increased GI ulcer risk"},
    ("ibuprofen", "metoprolol"): {"severity": "moderate", "effect": "Reduced BP control"},
    ("diclofenac", "losartan"): {"severity": "high", "effect": "Acute kidney injury risk"},
    ("azithromycin", "atorvastatin"): {"severity": "moderate", "effect": "Increased statin toxicity"},
    ("doxycycline", "calcium carbonate"): {"severity": "high", "effect": "Reduced doxycycline absorption"},
    ("metronidazole", "alcohol"): {"severity": "high", "effect": "Disulfiram-like reaction"},
    ("metformin", "prednisolone"): {"severity": "high", "effect": "Poor glucose control"},
    ("metformin", "contrast dye"): {"severity": "high", "effect": "Lactic acidosis risk"},
    ("glimepiride", "nsaid"): {"severity": "moderate", "effect": "Severe hypoglycemia"},
    ("sitagliptin", "ace inhibitor"): {"severity": "moderate", "effect": "Increased hypoglycemia risk"},
    ("omeprazole", "clopidogrel"): {"severity": "high", "effect": "Reduced clopidogrel effect"},
    ("pantoprazole", "iron folic acid"): {"severity": "moderate", "effect": "Reduced iron absorption"},
    ("ranitidine", "ketoconazole"): {"severity": "high", "effect": "Reduced antifungal effect"},
    ("warfarin", "aspirin"): {"severity": "high", "effect": "Severe bleeding risk"},
    ("warfarin", "nsaid"): {"severity": "high", "effect": "GI bleeding"},
    ("warfarin", "azithromycin"): {"severity": "high", "effect": "Increased INR"},
    ("paracetamol", "warfarin"): {"severity": "moderate", "effect": "Increased bleeding risk"},
    ("morphine", "benzodiazepine"): {"severity": "high", "effect": "Respiratory depression"},
    ("diclofenac", "ace inhibitor"): {"severity": "high", "effect": "Acute renal failure"},
}

# Add reverse pairs
for (drug1, drug2), info in list(INTERACTIONS.items()):
    INTERACTIONS[(drug2, drug1)] = info

# ============================================================================ 
# DRUG ALTERNATIVES
# ============================================================================ 
ALTERNATIVES = {
    "aspirin": ["paracetamol", "ibuprofen", "diclofenac"],
    "ibuprofen": ["paracetamol", "naproxen", "diclofenac", "etoricoxib"],
    "diclofenac": ["ibuprofen", "naproxen", "celecoxib"],
    "paracetamol": ["ibuprofen", "aspirin", "tramadol"],
    "metformin": ["sitagliptin", "dapagliflozin", "pioglitazone"],
    "pantoprazole": ["omeprazole", "esomeprazole"],
    "amoxicillin": ["cephalexin", "azithromycin"],
    "azithromycin": ["amoxicillin", "doxycycline"],
    "warfarin": ["clopidogrel"],
    "tramadol": ["paracetamol", "ibuprofen", "morphine"],
    "morphine": ["tramadol"],
    "ondansetron": ["domperidone"],
    "cetirizine": ["loratadine", "fexofenadine"],
    "acyclovir": ["oseltamivir"],
}

CONDITIONS = {
    "diabetes": {"avoid": ["prednisolone", "dexamethasone", "corticosteroid"], "reason": "Raises blood sugar"},
    "hypertension": {"avoid": ["nsaid", "decongestants"], "reason": "Raises blood pressure"},
    "asthma": {"avoid": ["aspirin", "nsaid", "beta_blocker"], "reason": "May trigger attack"},
    "kidney disease": {"avoid": ["nsaid", "metformin", "ace_inhibitor"], "reason": "Kidney damage"},
    "liver disease": {"avoid": ["paracetamol", "acetaminophen"], "reason": "Liver toxicity"},
    "pregnancy": {"avoid": ["warfarin", "ace_inhibitor", "finasteride"], "reason": "Birth defects"},
}

# ============================================================================ 
# DOSAGES (partial example, add remaining as needed)
# ============================================================================ 
DOSAGES = {
    "amoxicillin": {
        "child": {"dose": "20-40 mg/kg/day", "max": "90 mg/kg/day", "warning": "Divide into 2-3 doses, use under pediatric supervision"},
        "adult": {"dose": "250-500 mg every 8 hours", "max": "3000 mg/day", "warning": "Take with food to reduce GI upset"},
        "elderly": {"dose": "250-500 mg every 12 hours", "max": "3000 mg/day", "warning": "Monitor renal function, adjust if CrCl <30"}
    },
    "azithromycin": {
        "child": {"dose": "10 mg/kg on day 1, then 5 mg/kg", "max": "500 mg/day", "warning": "Single daily dose, complete course"},
        "adult": {"dose": "500 mg day 1, then 250 mg", "max": "500 mg/day", "warning": "Take on empty stomach"},
        "elderly": {"dose": "500 mg day 1, then 250 mg", "max": "500 mg/day", "warning": "Check for QT prolongation risk"}
    },
    "paracetamol": {
        "child": {"dose": "10-15 mg/kg/dose", "max": "60 mg/kg/day", "warning": "Every 4-6 hours, hepatotoxicity risk if overdose"},
        "adult": {"dose": "500-1000 mg every 4-6 hours", "max": "4000 mg/day", "warning": "Do not exceed max dose - liver damage"},
        "elderly": {"dose": "500 mg every 6 hours", "max": "3000 mg/day", "warning": "Monitor liver function"}
    },
    "ibuprofen": {
        "child": {"dose": "5-10 mg/kg/dose", "max": "40 mg/kg/day", "warning": "Every 6-8 hours with food"},
        "adult": {"dose": "200-400 mg every 6-8 hours", "max": "3200 mg/day", "warning": "Take with food, GI and kidney risk"},
        "elderly": {"dose": "200 mg every 8-12 hours", "max": "1600 mg/day", "warning": "Increased GI bleeding risk"}
    },
}

# ============================================================================ 
# OCR FUNCTIONS
# ============================================================================ 
def extract_text_easyocr(img: Image.Image) -> str:
    if not EASYOCR_AVAILABLE or OCR_READER is None:
        return ""
    try:
        img_np = np.array(img)
        results = OCR_READER.readtext(img_np)
        text = ' '.join([res[1] for res in results])
        return text.strip()
    except Exception as e:
        st.error(f"EasyOCR Error: {str(e)}")
        return ""

def preprocess_image_ultra_advanced(img: Image.Image) -> Image.Image:
    img_np = np.array(img.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    result = Image.fromarray(morph)
    w, h = result.size
    if w < 1024:
        new_w = 3000
        new_h = int(h * (new_w / w))
        result = result.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return result

def extract_text_tesseract_enhanced(img: Image.Image) -> str:
    if not PYTESSERACT_OCR_AVAILABLE:
        return None
    try:
        processed1 = preprocess_image_ultra_advanced(img)
        config1 = r'--oem 3 --psm 6 -l eng'
        text1 = pytesseract.image_to_string(processed1, config=config1)
        if text1.strip():
            return text1.strip()
        config2 = r'--oem 3 --psm 11 -l eng'
        text2 = pytesseract.image_to_string(processed1, config=config2)
        return text2.strip() if text2.strip() else None
    except Exception:
        return None

def extract_text_with_ocr(img: Image.Image) -> str:
    if EASYOCR_AVAILABLE and OCR_READER is not None:
        text = extract_text_easyocr(img)
        if text:
            return text
    if PYTESSERACT_OCR_AVAILABLE:
        text = extract_text_tesseract_enhanced(img)
        if text:
            return text
    return ""

def find_drugs_super_flexible(text: str) -> list:
    if not text:
        return []
    found = []
    text_lower = text.lower()
    seen = set()
    for drug_name in DRUGS.keys():
        if drug_name in seen:
            continue
        patterns = [r'\b' + re.escape(drug_name.lower()) + r'\b', r'\b' + re.escape(drug_name.lower()), re.escape(drug_name.lower())]
        found_drug = False
        for pattern in patterns:
            if re.search(pattern, text_lower):
                found_drug = True
                break
        if found_drug:
            dosage_patterns = [
                rf'{re.escape(drug_name.lower())}\s*[:\-]*\s*(\d+(?:\.\d+)?)\s*(mg|ml|mcg|units|gm|g|tab|tabs|cap|caps|unit)?',
                rf'(\d+(?:\.\d+)?)\s*(mg|ml|mcg|units|gm|g|tab|tabs|cap|caps)\s*{re.escape(drug_name.lower())}',
            ]
            dosage = "Not specified"
            for dosage_pattern in dosage_patterns:
                match = re.search(dosage_pattern, text_lower)
                if match:
                    dosage = match.group(0)
                    break
            found.append({'drug': drug_name, 'dosage': dosage})
            seen.add(drug_name)
    return found

def check_interactions(drug_list: list) -> list:
    results = []
    n = len(drug_list)
    for i in range(n):
        for j in range(i + 1, n):
            d1 = drug_list[i]['drug']
            d2 = drug_list[j]['drug']
            info = INTERACTIONS.get((d1, d2))
            if info:
                results.append({
                    'drug1': d1,
                    'drug2': d2,
                    'severity': info['severity'],
                    'effect': info['effect']
                })
    return results

def check_safety_conditions(drugs: list, conditions: list) -> list:
    warnings = []
    for cond in conditions:
        avoid_list = CONDITIONS.get(cond, {}).get('avoid', [])
        reason = CONDITIONS.get(cond, {}).get('reason', '')
        for drug_entry in drugs:
            drug = drug_entry['drug']
            drug_class = DRUGS.get(drug, {}).get('class')
            if drug in avoid_list or drug_class in avoid_list:
                warnings.append(f"Drug {drug} should be avoided in {cond} ({reason})")
    return warnings

def get_dosage_info(drug_name: str, age_category: str) -> dict:
    return DOSAGES.get(drug_name.lower(), {}).get(age_category, {"dose": "Unknown", "max": "Unknown", "warning": ""})

def get_alternatives(drug_name: str) -> list:
    return ALTERNATIVES.get(drug_name.lower(), [])

# ============================================================================ 
# STREAMLIT APP
# ============================================================================ 
st.set_page_config(page_title="Medical Prescription Analyzer", page_icon="💊", layout="wide")

st.title("💊 Medical Prescription Analyzer")
st.markdown("Upload a prescription image, check drug interactions, dosages, and safety warnings.")

tabs = ["Prescription OCR", "Interaction Checker", "Dosage Info", "Safety Check"]
tab_selected = st.tabs(tabs)

# ---------------------------------------------
# PRESCRIPTION OCR TAB
# ---------------------------------------------
with tab_selected[0]:
    st.header("📸 Prescription OCR")
    uploaded_file = st.file_uploader("Upload prescription image (jpg, png)", type=["jpg", "jpeg", "png"])
    age = st.number_input("Patient age (years)", min_value=0, max_value=120, value=30)

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Analyze Prescription"):
            text = extract_text_with_ocr(image)
            st.subheader("Extracted Text")
            st.write(text)

            drugs_found = find_drugs_super_flexible(text)
            if drugs_found:
                st.subheader("Detected Drugs & Dosages")
                for entry in drugs_found:
                    drug = entry['drug']
                    dosage = entry['dosage']
                    age_cat = 'child' if age < 12 else 'adult' if age < 65 else 'elderly'
                    dose_info = get_dosage_info(drug, age_cat)
                    st.markdown(f"**{drug.title()}** - {dosage} | Recommended: {dose_info['dose']} (Max: {dose_info['max']})")
                    if dose_info['warning']:
                        st.warning(dose_info['warning'])

                interactions = check_interactions(drugs_found)
                if interactions:
                    st.subheader("⚠️ Potential Interactions")
                    for inter in interactions:
                        st.error(f"{inter['drug1']} + {inter['drug2']} -> {inter['severity'].title()} Risk: {inter['effect']}")
                else:
                    st.success("No major interactions detected")
            else:
                st.warning("No drugs detected")

# ---------------------------------------------
# INTERACTION CHECKER TAB
# ---------------------------------------------
with tab_selected[1]:
    st.header("🔗 Drug Interaction Checker")
    drugs_input = st.text_area("Enter drug names (comma or newline separated)")
    if st.button("Check Interactions", key="check_interactions"):
        drugs_list = [d.strip() for d in re.split(r'[,\n]+', drugs_input) if d.strip()]
        drugs_structured = [{'drug': d.lower(), 'dosage': 'N/A'} for d in drugs_list if d.lower() in DRUGS]
        if drugs_structured:
            interactions = check_interactions(drugs_structured)
            if interactions:
                for inter in interactions:
                    st.error(f"{inter['drug1']} + {inter['drug2']} -> {inter['severity'].title()} Risk: {inter['effect']}")
            else:
                st.success("No interactions detected")
        else:
            st.warning("No recognized drugs entered")

# ---------------------------------------------
# DOSAGE INFO TAB
# ---------------------------------------------
with tab_selected[2]:
    st.header("📏 Dosage Information")
    drug_choice = st.selectbox("Select Drug", sorted(DRUGS.keys()))
    patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=30, key="dosage_age")
    if st.button("Get Dosage", key="get_dosage"):
        age_cat = 'child' if patient_age < 12 else 'adult' if patient_age < 65 else 'elderly'
        info = get_dosage_info(drug_choice, age_cat)
        st.markdown(f"**{drug_choice.title()}** - Recommended: {info['dose']} (Max: {info['max']})")
        if info['warning']:
            st.warning(info['warning'])
        alternatives = get_alternatives(drug_choice)
        if alternatives:
            st.info("Alternative drugs: " + ", ".join(alternatives))

# ---------------------------------------------
# SAFETY CHECK TAB
# ---------------------------------------------
with tab_selected[3]:
    st.header("🏥 Safety Check")
    condition_list = st.multiselect("Patient Medical Conditions", sorted(CONDITIONS.keys()))
    drugs_input2 = st.text_area("Current Medications (comma or newline separated)", key="safety_drugs")
    if st.button("Check Safety", key="safety_check"):
        drugs_list2 = [d.strip() for d in re.split(r'[,\n]+', drugs_input2) if d.strip()]
        drugs_structured2 = [{'drug': d.lower(), 'dosage': 'N/A'} for d in drugs_list2 if d.lower() in DRUGS]
        if drugs_structured2:
            safety_warnings = check_safety_conditions(drugs_structured2, condition_list)
            if safety_warnings:
                for warn in safety_warnings:
                    st.error(warn)
            else:
                st.success("No safety warnings for selected conditions")
        else:
            st.warning("No recognized drugs entered for safety check")
