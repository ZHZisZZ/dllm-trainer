import re, math
from fractions import Fraction
from decimal import Decimal, InvalidOperation
from typing import Optional

class Gsm8kScorer:
    # extract pred answer
    ANS_LINE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
    HINT_ANS = re.compile(r"(?i)\b(?:the\s+)?(?:final\s+)?answer\s*(?:is\s+equal\s+to|is|=|equals|:)\s*([^\n]+)")
    _BOXED = re.compile(r"\\boxed\{([^}]*)\}")
    NUM_FINDER = re.compile(r"[-+]?\d+(?:\.\d+)?")
    _MIXED_FRAC = re.compile(r"[-+]?\d+\s+\d+/\d+")
    _PURE_FRAC  = re.compile(r"[-+]?\d+/\d+")
    _DEC_SCI    = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?")
    # find gold answer
    _GSM_GOLD = re.compile(r"####\s*([^\n]+)")
    # truncate tail part of the generation
    _TRAIL_PUNCT = re.compile(r"[)\].,;:\s]+$")
    _LEAD_PUNCT  = re.compile(r"^[([{\s]+")
    _UNIT_TAIL = re.compile(
        r"\s*(dollars?|cents?|percent|perc\.?|pts?|points?|years?|year|hrs?|hours?|mins?|minutes?|secs?|seconds?)\s*$",
        re.IGNORECASE,
    )

    def __init__(self, atol: float = 1e-6):
        self.atol = float(atol)

    def _strip_wrappers(self, s: str) -> str:
        s = self._LEAD_PUNCT.sub("", s)
        s = self._TRAIL_PUNCT.sub("", s)
        return s.strip()

    def _simple_moneylike_last(self, text: str) -> str:
        m = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
        if not m:
            return ""
        last = m[-1]
        tok = [x for x in last if x][0]
        tok = tok.strip()
        for rgx in (",", r"\$", r"(?s).*#### ", r"\.$"):
            tok = re.sub(rgx, "", tok)
        return tok.strip()

    def _find_last_numberish(self, text: str) -> str:
        for rx in (self._MIXED_FRAC, self._PURE_FRAC, self._DEC_SCI):
            hits = list(rx.finditer(text))
        if hits:
            tok = hits[-1].group(0)
            return self._post_clean_numberish(tok)
        return self._simple_moneylike_last(text) or ""

    def _post_clean_numberish(self, s: str) -> str:
        if not s:
            return ""
        t = s.strip()
        boxed = self._BOXED.search(t)
        if boxed:
            t = boxed.group(1)
        for rx in (self._MIXED_FRAC, self._PURE_FRAC, self._DEC_SCI):
            hits = rx.findall(t)
            if hits:
                cand = hits[-1] if isinstance(hits, list) else hits
                t = cand if isinstance(cand, str) else cand[-1]
                break
        t = t.replace("$", "").replace(",", "")
        t = self._UNIT_TAIL.sub("", t)
        t = self._strip_wrappers(t)
        t = re.sub(r"%\s*$", "", t)
        return t.strip()
    
    def _parse_numeric(self, s: str):
        if s is None:
            return None
        t = str(s).strip()
        if not t:
            return None
        m = re.fullmatch(r"([+-]?\d+)\s+(\d+)\s*/\s*(\d+)", t)
        if m:
            whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
            sign = -1 if whole < 0 else 1
            whole = abs(whole)
            try:
                return Fraction(whole * den + num, den) * sign
            except ZeroDivisionError:
                return None
        m = re.fullmatch(r"([+-]?\d+)\s*/\s*(\d+)", t)
        if m:
            try:
                return Fraction(int(m.group(1)), int(m.group(2)))
            except ZeroDivisionError:
                return None
        try:
            v = float(Decimal(t))
            return v
        except (InvalidOperation, ValueError):
            return None
        
    # ---------------- public API ----------------
    def normalize_number(self, x: str) -> str:
        if x is None:
            return ""
        s = str(x).strip()
        s = self._strip_wrappers(s)
        s = s.replace(",", "").replace("$", "").strip()
        s = re.sub(r"\s*%\s*$", "", s)
        s = re.sub(r"\s*percent\s*$", "", s, flags=re.IGNORECASE)
        m = re.fullmatch(r"\s*([+-]?\d+)\s+(\d+)\s*/\s*(\d+)\s*\Z", s)
        if m:
            whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
            sign = -1 if whole < 0 else 1
            whole = abs(whole)
            try:
                frac = Fraction(whole * den + num, den) * sign
                return f"{frac.numerator}/{frac.denominator}"
            except ZeroDivisionError:
                return ""
        m = re.fullmatch(r"\s*([+-]?\d+)\s*/\s*(\d+)\s*\Z", s)
        if m:
            try:
                frac = Fraction(int(m.group(1)), int(m.group(2)))
                return f"{frac.numerator}/{frac.denominator}"
            except ZeroDivisionError:
                return ""
        try:
            v = float(Decimal(s))
            if math.isfinite(v) and abs(v - round(v)) < 1e-12:
                return str(int(round(v)))
            return f"{v:.12g}"
        except (InvalidOperation, ValueError):
            return s
        
    def extract_gold(self, gold_field: str) -> str:
        if not gold_field:
            return ""
        m = self._GSM_GOLD.search(gold_field)
        ans = (m.group(1) if m else gold_field).strip()
        return self._strip_wrappers(ans)
    
    def extract_pred(self, text: str) -> str:
        if not text:
            return ""
        m_all = list(self._BOXED.finditer(text))
        if m_all:
            cand = self._post_clean_numberish(m_all[-1].group(1))
            if cand:
                return cand
        m = self.ANS_LINE.search(text)
        if m:
            cand = self._post_clean_numberish(m.group(1))
            if cand:
                return cand
        m2 = self.HINT_ANS.search(text)
        if m2:
            cand = self._post_clean_numberish(m2.group(1))
            if cand:
                return cand
        last = self._find_last_numberish(text)
        return last or ""
    
    def grade(self, pred: str, gold: str) -> bool:
        p_norm = self.normalize_number(pred)
        g_norm = self.normalize_number(gold)
        if p_norm == g_norm:
            return True
        p_val = self._parse_numeric(p_norm)
        g_val = self._parse_numeric(g_norm)
        if p_val is None or g_val is None:
            return False
        if isinstance(p_val, Fraction) and isinstance(g_val, Fraction):
            return p_val == g_val
        try:
            pv = float(p_val) if not isinstance(p_val, float) else p_val
            gv = float(g_val) if not isinstance(g_val, float) else g_val
            return math.isfinite(pv) and math.isfinite(gv) and abs(pv - gv) <= self.atol
        except Exception:
            return False


GSM8KScorer = Gsm8kScorer(atol=1e-6)