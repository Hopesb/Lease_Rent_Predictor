"""Shared feature code. MUST match the training notebook exactly (the saved model depends on it)."""
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

STREET_TYPES = r'\b(STREET|ST|CRESCENT|CRES|CLOSE|ROAD|RD|AVENUE|AVE|DRIVE|DR|LANE|WAY|BOULEVARD|BLVD|HIGHWAY|EXPRESSWAY|TERRACE|PLACE|PLAZA|SQUARE|QUARTERS)\b'
ESTATE_KW = r'\b(ESTATE|GARDENS?|COURT|TOWERS?|CONDOMINIUM|MAISONETTES?|HOUSING|LAYOUT|SCHEME|PHASE|VIEW|PARK|MEADOWS?|SPRING|CREST|GRA)\b'
UNIT_KW = {
    'shop': r'\bSHOP\b',
    'self_contained': r'\bSELF[\s-]?CON(TAINED)?\b',
    'mini_flat': r'\bMINI[\s-]?FLAT\b',
    'flat': r'\bFLAT\b',
    'duplex': r'\bDUPLEX\b',
    'bungalow': r'\bBUNGALOW\b',
    'terrace': r'\bTERRACE\b',
    'maisonette': r'\bMAISONETTE\b',
    'single_room': r'\bSINGLE\s*ROOM\b|\bROOM\b',
    'church': r'\bCHURCH\b',
}
BEDROOM_RE = r'(\d+)\s*(?:BEDROOM|BED\s*ROOM|BR)\b'
FLOOR_RE = r'(\d+)(?:ST|ND|RD|TH)?\s*FLOOR'
BQ_RE = r'\bBQ\b|\bBOYS?\s*QUARTERS?\b'
FURNISHED_RE = r'\bFURNISHED\b|\bSERVICED\b'
NIGERIAN_STATES = {'LAGOS', 'OGUN', 'OYO', 'OSUN', 'ONDO', 'EKITI', 'KWARA', 'KOGI', 'ABUJA',
                   'FCT', 'RIVERS', 'KANO', 'KADUNA', 'ENUGU', 'ANAMBRA', 'DELTA', 'EDO', 'ABIA', 'IMO', 'BAYELSA'}


def parse_property(raw_addr):
    a = str(raw_addr).upper()
    a = re.sub(r'\b(N0|NO\.?)\s*', 'NO ', a)
    parts = [p.strip() for p in a.split(',') if p.strip()]
    out = {
        'n_segments': len(parts),
        'has_number': bool(re.search(r'\b\d+\b', a)),
        'has_estate': bool(re.search(ESTATE_KW, a)),
        'street_type': None, 'street_name': None, 'locality': None, 'city': None, 'state': None,
        'unit_type': None, 'bedrooms': None, 'floor': None,
        'has_bq': bool(re.search(BQ_RE, a)),
        'is_furnished': bool(re.search(FURNISHED_RE, a)),
    }
    m = re.search(STREET_TYPES, a)
    if m:
        out['street_type'] = m.group(1)
    for p in parts:
        if re.search(STREET_TYPES, p):
            out['street_name'] = p
            break
    if parts:
        last = parts[-1]
        tok = last.replace(' STATE', '').strip()
        if tok in NIGERIAN_STATES or 'STATE' in last:
            out['state'] = tok
            if len(parts) >= 2:
                out['city'] = parts[-2]
    if out['city'] and len(parts) >= 3:
        out['locality'] = parts[-3]
    for name, pat in UNIT_KW.items():
        if re.search(pat, a):
            out['unit_type'] = name
            break
    mb = re.search(BEDROOM_RE, a)
    if mb:
        out['bedrooms'] = int(mb.group(1))
    mf = re.search(FLOOR_RE, a)
    if mf:
        out['floor'] = int(mf.group(1))
    return out


def last_n(addr, n=3):
    parts = [p.strip() for p in str(addr).upper().split(',') if p.strip()]
    return ', '.join(parts[-n:])


def _to_num(s):
    """Parse 1,000,000 / 1.500.000 / 1.5M / 750k into a float (None if unparseable)."""
    if not s:
        return None
    s = s.strip().replace(' ', '')
    mult = 1
    if s.endswith(('M', 'm')):
        mult, s = 1_000_000, s[:-1]
    if s.endswith(('K', 'k')):
        mult, s = 1_000, s[:-1]
    s = s.replace(',', '').replace('.', '') if s.count('.') > 1 else s.replace(',', '')
    try:
        return float(s) * mult
    except ValueError:
        return None


NOTE_FLAGS = {
    'note_quit': r'NOTICE TO QUIT',
    'note_move_out': r'(?:TO MOVE OUT|VACATE|TO LEAVE BY)',
    'note_non_renewal': r'NON[\s-]?RENEWAL',
    'note_review_flag': r'RENT SHOULD BE REVIEWED',
    'note_payment_plan': r'(?:EXPECTED TO PAY|PLEAD TO PAY|PAY \d+\s*MONTHS?\s*RENT)',
    'note_enforcement': r'BREAK OPEN|GO TO THE APARTMENT',
    'note_malformed': r'malformed',
    'note_confirm': r'\bconfirm\b',
    'note_blank': r'pending confirmation|blank',
    'note_missing_src': r'Source S/N missing',
    'note_contact_fail': r'not reachable|not rechable',
    'note_dq_assumed': r'assumed|calculated as|set to month-end',
    'note_struct_shop': r'structure type:\s*SHOP',
    'note_struct_apt': r'structure type:\s*APARTMENT',
    'note_type_conflict': r'conflicting SHOP / APARTMENT',
    'note_multi_shop': r'SHOP ONE',
    'note_multi_unit': r'TWO (?:SHOP|UNITS?)',
    'note_estate_mgmt': r'ESTATE MANAGEMENT',
    'note_locality_varies': r'locality varies',
    'note_addr_ambig': r'(?:mentions both|combines)',
    'note_six_month': r'\(Six Months\)',
    'note_each_year': r'EACH YEAR',
    'note_furnished': r'FURNISHED|SERVICED',
    'note_bq': r'\bBQ\b|BOYS?\s*QUARTERS?',
    'note_flower': r'FLOWER MAINTENANCE',
}

CAT_COLS = ['property_name', 'street_type', 'street_name', 'locality', 'city', 'state',
            'unit_type', 'addr_last3', 'addr_last2']
BASE_NUM_COLS = ['n_segments', 'has_number', 'has_estate', 'has_bq', 'is_furnished', 'bedrooms',
                 'floor', 'service_charge', 'lease_months', 'start_year', 'start_month']
TEXT_COL = 'notes'
AUX_COLS = ['note_rent_val']   # used ONLY to repair the target, never as a model input


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw lease rows -> model-ready feature table (+ aux columns for target repair)."""
    d = raw.copy()
    notes = d['notes'].fillna('').astype(str)

    parsed = pd.DataFrame([parse_property(a) for a in d['property_name']], index=d.index)
    d = pd.concat([d, parsed], axis=1)
    d['addr_last3'] = d['property_name'].apply(last_n)
    d['addr_last2'] = d['property_name'].apply(lambda x: last_n(x, 2))

    for k, pat in NOTE_FLAGS.items():
        d[k] = notes.str.contains(pat, case=False, regex=True).astype(int)

    def grab_rent(text):
        m = re.search(r'Original rent details:\s*([\d,\.]+(?:[MmKk])?)', text, re.IGNORECASE)
        return _to_num(m.group(1)) if m else None
    d['note_rent_val'] = pd.to_numeric(notes.apply(grab_rent), errors='coerce')

    start = pd.to_datetime(d['start_date'], errors='coerce')
    end = pd.to_datetime(d['end_date'], errors='coerce')
    d['lease_months'] = ((end - start).dt.days / 30.4375).round()
    d['start_year'] = start.dt.year
    d['start_month'] = start.dt.month

    for c in ['has_number', 'has_estate', 'has_bq', 'is_furnished']:
        d[c] = d[c].astype(int)
    for c in ['bedrooms', 'floor', 'service_charge', 'lease_months', 'start_year', 'start_month']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    for c in CAT_COLS:
        d[c] = d[c].astype(object).where(d[c].notna(), 'missing').astype(str)
    d['notes'] = notes
    return d




class NotesSVD(BaseEstimator, TransformerMixin):
    """TF-IDF (1-2 grams) -> TruncatedSVD on the free-text notes. Fitted on training folds only."""

    def __init__(self, n_components=15, max_features=300, min_df=2, random_state=0):
        self.n_components = n_components
        self.max_features = max_features
        self.min_df = min_df
        self.random_state = random_state

    @staticmethod
    def _prep(X):
        s = pd.Series(np.asarray(X).ravel()).fillna('').astype(str)
        s = s.str.replace(r'Original (rent details|street address / unit description|structure type):', '',
                          regex=True)
        s = s.str.replace(r'[\d,]+', ' ', regex=True)   # amounts are not text signal
        return s.str.lower()

    def fit(self, X, y=None):
        txt = self._prep(X)
        self.tfidf_ = TfidfVectorizer(ngram_range=(1, 2), min_df=self.min_df, max_features=self.max_features)
        M = self.tfidf_.fit_transform(txt)
        k = max(1, min(self.n_components, M.shape[1] - 1, M.shape[0] - 1))
        self.svd_ = TruncatedSVD(n_components=k, random_state=self.random_state).fit(M)
        return self

    def transform(self, X):
        return self.svd_.transform(self.tfidf_.transform(self._prep(X)))


