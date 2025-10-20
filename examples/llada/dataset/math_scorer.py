from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple
import sympy as sp
from sympy.parsing.latex import parse_latex
import numpy as np
import re, math


class MathScorer:
    # ----------------------------- regexes -----------------------------
    _UNITS_RX = re.compile(r"\\text\{[^}]*\}\s*$")
    _DEG_RX   = re.compile(r"\^\{?\\?circ\}?|°")
    _MOD_TAIL = re.compile(r"\s*(?:\\?mod|\\bmod)\s*[0-9]+\s*$", re.IGNORECASE)
    _UNIT_WORD_TAIL = re.compile(r"\s*[A-Za-z][A-Za-z.%\-]*(?:\s+[A-Za-z][A-Za-z.%\-]*)*\s*$")

    _RATIO_RX = re.compile(r"^\s*([+-]?\d+)\s*:\s*([+-]?\d+)\s*$")
    _ABS_LATEX = re.compile(r"\\left\|([^|]+)\\right\|")

    _LATEX_STRIP = [
        (re.compile(r"\\left\s*"), ""),
        (re.compile(r"\\right\s*"), ""),
        (re.compile(r"\\!"), ""),
        (re.compile(r"\\mathrm\{([^}]*)\}"), r"\1"),
        (re.compile(r"\\operatorname\{([^}]*)\}"), r"\1"),
        (re.compile(r"\\text\{([^}]*)\}"), r"\1"),
    ]

    ANS_LINE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
    HINT_ANS = re.compile(r"(?i)\b(?:the\s+)?(?:final\s+)?answer\s*(?:is\s+equal\s+to|is|=|equals|:)\s*([^\n]+)")

    _LEAD_PUNCT  = re.compile(r"^[([\{\s]+")
    _TRAIL_PUNCT = re.compile(r"[)\].,;:\s]+$")

    # \boxed, \boxed*, \fbox, \bbox 모두 지원
    BOXED_OPEN = re.compile(r"\\boxed\*?\s*\{|\s\\fbox\s*\{|\s\\bbox\s*\{")

    # intervals/sets
    IN_SYMB = re.compile(r"\\?in\b")
    INTERVAL_RX = re.compile(r"([\[(])\s*([^,]+?)\s*,\s*([^)\]]+?)\s*([)\]])")
    SET_BRACES_RX = re.compile(r"^\s*\{(.+)\}\s*$")
    CSV_SIMPLE_RX = re.compile(r"^\s*[-+]?\d+(\s*,\s*[-+]?\d+)+\s*$")

    # numbers
    DEC_SCI = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?|[-+]?\.\d+(?:[eE][+-]?\d+)?")
    MIXED_FRAC = re.compile(r"([+-]?\d+)\s+(\d+)\s*/\s*(\d+)")
    PURE_FRAC  = re.compile(r"([+-]?\d+)\s*/\s*(\d+)")

    NO_SOLUTION_TOKENS = {"no solution", "nosolution", "none", "empty set", "emptyset", "∅", "varnothing", "\\varnothing", "{}"}
    DNE_TOKENS = {"dne", "does not exist", "undefined", "not defined"}
    INF_TOKENS = {"infinity", "\\infty", "∞"}

    # tolerant container helpers
    _SPACE_AFTER_COMMA = re.compile(r",\s*")
    _MULTI_SPACE = re.compile(r"\s+")
    _BARE_INF = re.compile(r"(?<!\\)inf(?:inity)?|∞|oo|\\infty", re.I)

    _ASSIGN_RE = re.compile(r"^\s*[a-zA-Z]\s*=\s*(.+)$")
    _MAT_RX = re.compile(r"\\begin\{pmatrix\}(.+?)\\end\{pmatrix\}", re.DOTALL)

    def __init__(self, atol: float = 1e-6):
        self.atol = float(atol)

    # ----------------------------- helpers -----------------------------
    # punctuation deleted: e.g. [x+y] -> "x+y"
    def _strip_wrappers(self, s: str) -> str:
        s = self._LEAD_PUNCT.sub("", s)
        s = self._TRAIL_PUNCT.sub("", s)
        return s.strip()

    # latex cleaning (unit text, \n, degree, etc.)
    def _basic_clean(self, s: str) -> str:
        if not s:
            return ""
        t = s.replace("\n", " ")
        # remove TeX wrappers/dollar
        t = re.sub(r"\$+", "", t)
        t = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", t)
        # t = t.replace("\\\\", "\\").replace("\\$", "$")
        t = t.replace("\\$", "$")
        t = re.sub(r"\\(?![A-Za-z])", "", t)
        for rx, rep in self._LATEX_STRIP:
            t = rx.sub(rep, t)
        t = self._UNITS_RX.sub("", t)
        t = self._DEG_RX.sub("", t)
        t = t.replace("%", "")
        t = re.sub(r"\s+", " ", t).strip()
        return t

    # 중첩 box 허용: e.g. \boxed{\frac{1}{2}} -> "1/2"
    def _unbox_last(self, text: str) -> Optional[str]:
        if not text:
            return None
        i = 0
        payload = None
        while True:
            m = self.BOXED_OPEN.search(text, i)
            if not m:
                break
            j = m.end()
            depth = 1
            buf: List[str] = []
            while j < len(text) and depth > 0:
                ch = text[j]
                if ch == '\\':
                    if j + 1 < len(text):
                        buf.append(text[j:j+2])
                        j += 2
                        continue
                if ch == '{':
                    depth += 1
                    buf.append(ch)
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                    buf.append(ch)
                else:
                    buf.append(ch)
                j += 1
            if depth == 0:
                payload = ''.join(buf)
                i = j
            else:
                break
        if payload is None:
            return None
        s = self._fix_fracs_sqrt(payload)
        # s = self._strip_wrappers(s)
        s = self._basic_clean(s)
        return s.strip()

    def _fix_fracs_sqrt(self, s: str) -> str:
        def _fix_fracs(u: str) -> str:
            # \dfrac,\tfrac → \frac
            u = re.sub(r"\\[dt]frac\b", r"\\frac", u)
            # \frac{A}B  → \frac{A}{B}
            u = re.sub(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*([^\s{])",
                    r"\\frac{\1}{\2}", u)
            # \frac A{B} → \frac{A}{B}
            u = re.sub(r"\\frac\s*([^\s{])\s*\{\s*([^{}]+?)\s*\}",
                    r"\\frac{\1}{\2}", u)
            # \frac A B  → \frac{A}{B}   (양쪽이 한 글자일 때만 안전하게)
            u = re.sub(r"\\frac\s*([^\s{])\s*([^\s{])",
                    r"\\frac{\1}{\2}", u)
            return u
        
        def _fix_sqrt(u: str) -> str:
            # bare sqrt{...} → \sqrt{...}
            u = re.sub(r'(?<!\\)sqrt\s*\{', r'\\sqrt{', u)
            if "\\sqrt" not in u:
                return u
            parts = u.split("\\sqrt")
            out = parts[0]
            for tail in parts[1:]:
                if tail and tail[0] != "{":
                    out += "\\sqrt{" + tail[0] + "}" + tail[1:]
                else:
                    out += "\\sqrt" + tail
            return out

        return _fix_sqrt(_fix_fracs(s))

    def _expand_pm(self, expr: str) -> List[str]:
        if "±" not in expr:
            return [expr]
        a = expr.replace("±", "+", 1)
        b = expr.replace("±", "-", 1)
        if "±" in a or "±" in b:
            acc: List[str] = []
            acc.extend(self._expand_pm(a))
            acc.extend(self._expand_pm(b))
            return acc
        return [a, b]

    def _pre_extract_scalar(self, s: str) -> str:
        if not s:
            return ""
        b = self._unbox_last(s)
        if b:
            return b
        m = self.ANS_LINE.search(s)
        if m:
            return self._strip_wrappers(self._fix_fracs_sqrt(m.group(1)))
        m2 = self.HINT_ANS.search(s)
        if m2:
            return self._strip_wrappers(self._fix_fracs_sqrt(m2.group(1)))
        return s

    def _last_latex_token(self, s: str) -> Optional[str]:
        if not s:
            return None
        u = self._fix_fracs_sqrt(s)
        # 마지막 \frac{...}{...} 또는 \sqrt{...} 를 잡음
        patt = re.compile(r"(\\frac\s*\{\s*[^{}]+\s*\}\s*\{\s*[^{}]+\s*\}|\\sqrt\s*\{\s*[^{}]+\s*\})")
        hits = list(patt.finditer(u))
        if hits:
            tok = hits[-1].group(0)
            return self._strip_wrappers(self._basic_clean(tok))
        return None
    
    def _maybe_take_whole_expr(self, t: str) -> Optional[str]:
        u = t.strip()
        if len(u) <= 80 and re.search(r"[+\-*/^]", u) and ("/" in u or "\\" in u) and re.search(r"[A-Za-z]", u):
            return u
        return None
    
    # ------------------------- public: extract --------------------------
    def extract_pred(self, text: str) -> str:
        if not text:
            return ""
        # 1) most trustable: boxed
        b = self._unbox_last(text)
        if b:
            if self.IN_SYMB.search(b):
                m = self.INTERVAL_RX.search(b)
                if m:
                    return m.group(0)    #self._strip_wrappers(m.group(0))
            csv = self._csv_to_tuple_str(b)
            return csv or b
        # 2) explicit Answer: ...
        m = self.ANS_LINE.search(text)
        if m:
            return self._strip_wrappers(self._fix_fracs_sqrt(m.group(1)))
        m2 = self.HINT_ANS.search(text)
        if m2:
            return self._strip_wrappers(self._fix_fracs_sqrt(m2.group(1)))
        # latex token
        tok = self._last_latex_token(text)
        if tok:
            return tok
        # 3) last interval or last numeric-ish token
        t = self._basic_clean(text)
        whole = self._maybe_take_whole_expr(t)
        if whole:
            return whole
        parts = re.split(r"\s*\\cup\s*", t)
        if len(parts) > 1:
            ivs: List[str] = []
            ok = True
            for p in parts:
                p = p.strip()
                m = self.INTERVAL_RX.search(p)
                if not m:
                    ok = False
                    break
                ivs.append(self._strip_wrappers(m.group(0)))
            if ok and ivs:
                return r" \cup ".join(ivs)

        inter = self._extract_interval_str(t)
        if inter:
            return inter #self._strip_wrappers(inter)
        # fallback: grab last number-like or keep tail
        for rx in (self.MIXED_FRAC, self.PURE_FRAC, self.DEC_SCI):
            hits = list(rx.finditer(t))
            if hits:
                return self._strip_wrappers(hits[-1].group(0))
        return t.strip()

    def extract_gold(self, rec: Any) -> str:
        # 0) Prefer explicit solution fields -> MATH
        if isinstance(rec, dict):
            sol = ""
            # MATH 원본 + 표준화된 'answer'까지 모두 커버
            for k in ["solution","solutions","proof","gold_sol","solution_text","solution_html","answer"]:
                if rec.get(k):
                    sol = str(rec[k]); break
        else:
            sol = str(rec) if rec is not None else ""
        if not sol:
            return ""
        # 1) boxed wins
        b = self._unbox_last(sol)
        if b:
            if self.IN_SYMB.search(b):
                after_in = self.INTERVAL_RX.search(b)
                if after_in:
                    return after_in.group(0)  # self._strip_wrappers(after_in.group(0))
            csv = self._csv_to_tuple_str(b)
            return csv or b
        # 2) look for last explicit interval in solution
        t = self._basic_clean(sol)
        whole = self._maybe_take_whole_expr(t)
        if whole:
            return whole
        tok = self._last_latex_token(sol)
        if tok:
            return tok
        inter = self._extract_interval_str(t)
        if inter:
            return inter # self._strip_wrappers(inter)
        # 3) look for explicit Answer lines in solution
        m = self.ANS_LINE.search(sol)
        if m:
            return self._strip_wrappers(self._fix_fracs_sqrt(m.group(1)))
        m2 = self.HINT_ANS.search(sol)
        if m2:
            return self._strip_wrappers(self._fix_fracs_sqrt(m2.group(1)))
        # 4) bare CSV at tail
        csv = self._csv_to_tuple_str(t)
        if csv:
            return csv
        # 5) fallback: last number
        for rx in (self.MIXED_FRAC, self.PURE_FRAC, self.DEC_SCI):
            hits = list(rx.finditer(t))
            if hits:
                return hits[-1].group(0)
        return t.strip()

    # ---------------------- normalization utilities ---------------------
    NO_SOLUTION = "<NO_SOLUTION>"; DNE = "<DNE>"; INF = "<INF>"

    def _normalize_special_tokens(self, s: str) -> str:
        t = self._basic_clean(s).lower().strip()
        if t in self.NO_SOLUTION_TOKENS: return self.NO_SOLUTION
        if t in self.DNE_TOKENS: return self.DNE
        if t in self.INF_TOKENS: return self.INF
        return self._basic_clean(s)

    # Convert ratio a:b to a/b
    def _ratio_to_frac(self, s: str) -> Optional[str]:
        m = self._RATIO_RX.fullmatch(s)
        if not m:
            return None
        a, b = int(m.group(1)), int(m.group(2))
        if b == 0:
            return None
        f = Fraction(a, b)
        return f"{f.numerator}/{f.denominator}"

    def _strip_assignment(self, s: str) -> str:
        m = self._ASSIGN_RE.match(s)
        return m.group(1).strip() if m else s

    # various formats of fraction normalization
    def _normalize_atomic(self, x: str) -> str:
        if x is None:
            return ""
        s = self._pre_extract_scalar(x)
        s = self._strip_assignment(s)
        b = self._unbox_last(s)
        if b is not None:
            s = b
        s = self._strip_wrappers(s)
        s = self._MOD_TAIL.sub("", s)
        s = self._UNITS_RX.sub("", s)
        s = self._DEG_RX.sub("", s)
        # 숫자 + 영단어 꼬리 → 꼬리 제거 (순수 숫자일 때만)
        s_num_tail = self._UNIT_WORD_TAIL.sub("", s)
        if s_num_tail != s:
            if (re.fullmatch(r"\s*[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*\Z", s_num_tail)
                or re.fullmatch(r"\s*[-+]?\.\d+\s*\Z", s_num_tail)
                or re.fullmatch(r"\s*([+-]?\d+)\s*/\s*(\d+)\s*\Z", s_num_tail)
                or re.fullmatch(r"\s*\\frac\s*\{\s*[^{}]+\s*\}\s*\{\s*[^{}]+\s*\}\s*\Z", s_num_tail)
            ):
                s = s_num_tail
        # Turn ratio a:b into a/b
        r = self._ratio_to_frac(s)
        if r is not None:
            return r
        # Mixed fraction "a b/c" → improper
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
        # Pure fraction a/b
        m = re.fullmatch(r"\s*([+-]?\d+)\s*/\s*(\d+)\s*\Z", s)
        if m:
            try:
                frac = Fraction(int(m.group(1)), int(m.group(2)))
                return f"{frac.numerator}/{frac.denominator}"
            except ZeroDivisionError:
                return ""
        # Pure numeric possibly with thousands separators and/or decimals
        s_num_like = re.fullmatch(r"\s*[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*\Z", s)
        if s_num_like:
            try:
                v = float(Decimal(s.replace(",", "")))
                if math.isfinite(v) and abs(v - round(v)) < 1e-12:
                    return str(int(round(v)))
                return f"{v:.12g}"
            except (InvalidOperation, ValueError):
                pass
        return s

    def normalize_number(self, x: str) -> str:
        return self._normalize_atomic(x)

    # -------------------- sympy-friendly normalization ------------------
    def _balance_brackets(self, s: str) -> str:
        if not s:
            return s
        openers = {"(": ")", "[": "]", "{": "}"}
        closers = {")": "(", "]": "[", "}": "{"}
        stack: List[str] = []
        out: List[str] = []
        for ch in s:
            if ch in openers:
                stack.append(ch)
                out.append(ch)
            elif ch in closers:
                if stack and stack[-1] == closers[ch]:
                    stack.pop()
                    out.append(ch)
                else:
                    # drop unmatched closer
                    continue
            else:
                out.append(ch)
        # drop unmatched openers at end
        while out and stack:
            ch = out.pop()
            if ch in openers:
                stack.pop()
                continue
            out.append(ch)
            break
        return "".join(out)

    def _rebalance_parens(self, s: str) -> str:
        if not s:
            return s
        # 먼저 필요한 만큼 '('를 앞에 추가
        opens, closes = s.count("("), s.count(")")
        if closes > opens:
            s = "(" * (closes - opens) + s
        # 다시 세서 닫는 쪽도 맞춰줌
        opens, closes = s.count("("), s.count(")")
        if opens > closes:
            s = s + ")" * (opens - closes)
        return s
    
    def _latex_cleanup(self, expr: str) -> str:
        if not expr:
            return ""
        t = self._balance_brackets(self._basic_clean(expr))
        t = self._rebalance_parens(t)
        t = re.sub(r"\\[dt]frac", r"\\frac", t)
        t = self._fix_fracs_sqrt(t)
        # 1) pi latex
        t = t.replace("\\pi", "pi")
        # 2) \frac{a}{b} -> (a)/(b),  \sqrt{u} / sqrt{u} -> sqrt(u)
        t = re.sub(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", r"(\1)/(\2)", t)
        t = re.sub(r"(?:\\sqrt|(?<!\\)sqrt)\s*\{([^{}]+)\}", r"sqrt(\1)", t)
        # 3) ops
        t = t.replace("\\cdot", "*").replace("\\times", "*")
        t = t.replace("^", "**").replace("\\pm", "±")
        # 4) |x| (left/right), 조합
        t = self._ABS_LATEX.sub(lambda m: f"Abs({m.group(1)})", t)
        t = re.sub(r"\\binom\{([^}]*)\}\{([^}]*)\}", r"binomial(\1,\2)", t)
        t = re.sub(r"\{([^}]*)\}\\choose\{([^}]*)\}", r"binomial(\1,\2)", t)
        # 5) 암시적 곱셈 보강: 5sqrt(2), 2(x+1)
        t = re.sub(r'(\d)\s*(?=\\[A-Za-z])', r'\1*', t)
        t = re.sub(r'(\d)\s*(?=[A-Za-z(])', r'\1*', t)
        t = re.sub(r'(\))\s*(?=[A-Za-z(])', r'\1*', t)
        # t = re.sub(r'([A-Za-z])\s*(\()', r'\1*(', t)
        return t.strip()

    # string '[a,b]', '(a,b]' to SymPy Interval
    def _interval_to_sympy(self, s: str):
        m = self.INTERVAL_RX.search(s)
        if not m:
            return None
        L, a_raw, b_raw, R = m.groups()
        left_open  = (L == '(')
        right_open = (R == ')')

        def _parse_end(u: str):
            u = self._basic_clean(u)
            if u.lower() in {"-inf", "-infty", "-∞", "-oo", "-\\infty"}: return -sp.oo if sp else None
            if u.lower() in {"inf", "+inf", "infty", "∞", "oo", "\\infty"}: return sp.oo if sp else None
            try:
                if parse_latex is not None and re.search(r"\\[a-zA-Z]+", u):
                    return parse_latex(self._basic_clean(u))
            except Exception:
                pass
            try:
                return sp.sympify(self._latex_cleanup(u), rational=True) if sp else None
            except Exception:
                return None

        a = _parse_end(a_raw); b = _parse_end(b_raw)
        if a is None or b is None or sp is None:
            return None
        try:
            return sp.Interval(a, b, left_open=left_open, right_open=right_open)
        except Exception:
            return None

    def _sympy_parse(self, expr: str):
        if expr is None:
            return None
        raw = expr.strip()
        if not raw:
            return None
        # union of intervals
        parts = re.split(r"\s*\\cup\s*", raw)
        if len(parts) > 1 and sp is not None:
            ivs = []
            for p in parts:
                p = p.strip()
                iv = self._interval_to_sympy(p)
                if iv is None:
                    # not interval → 일반 expr 시도
                    try:
                        iv = sp.sympify(self._latex_cleanup(p), rational=True)
                    except Exception:
                        iv = None
                if iv is None:
                    return None
                ivs.append(iv)
            try:
                return sp.Union(*ivs)  # type: ignore[attr-defined]
            except Exception:
                return None
        # single interval
        iv = self._interval_to_sympy(raw)
        if iv is not None:
            return iv

        looks_latex = bool(re.search(r"\\[a-zA-Z]+", raw))
        try:
            if looks_latex and parse_latex is not None:
                return parse_latex(self._latex_cleanup(raw))
        except Exception:
            pass
        try:
            return sp.sympify(self._latex_cleanup(raw), rational=True) if sp else None
        except Exception:
            return None

    def _sympy_equal(self, a_str: str, b_str: str) -> bool:
        if sp is None:
            return False
        a = self._sympy_parse(a_str)
        b = self._sympy_parse(b_str)
        if a is None or b is None:
            return False
        try:
            # set/interval exact equality
            if isinstance(a, (sp.Set, sp.Interval)) or isinstance(b, (sp.Set, sp.Interval)):
                return sp.simplify(a) == sp.simplify(b)
        except Exception:
            pass
        try:
            if sp.simplify(a - b) == 0:
                return True
        except Exception:
            pass
        # heuristic sampling
        try:
            vars = sorted(list((a.free_symbols | b.free_symbols)), key=lambda s: s.name)  # type: ignore[attr-defined]
            if not vars:
                return False
            for val in [-2, -1, 0, 1, 2, 3]:
                subs = {v: val for v in vars}
                av = complex(a.evalf(subs=subs))  # type: ignore
                bv = complex(b.evalf(subs=subs))  # type: ignore
                if abs(av - bv) > 1e-8:
                    return False
            return True
        except Exception:
            return False

    # ------------------------ containers handling -----------------------
    def _extract_interval_str(self, s: str) -> Optional[str]:
        if not s:
            return None
        cand = None
        for m in self.INTERVAL_RX.finditer(s):
            cand = m.group(0)
        return cand

    # Bare CSV like "4, 6, 14, 15" -> "(4, 6, 14, 15)"
    def _csv_to_tuple_str(self, s: str) -> Optional[str]:
        t = self._basic_clean(s)
        if self.CSV_SIMPLE_RX.fullmatch(t):
            return f"({t})"
        return None

    def _split_top_level_commas(self, s: str) -> List[str]:
        items, buf, depth = [], [], 0
        for ch in s:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                items.append("".join(buf)); buf = []
            else:
                buf.append(ch)
        items.append("".join(buf))
        return [x.strip() for x in items if x.strip()]

    def _as_container(self, s: str) -> Tuple[Optional[str], List[str]]:
        t = self._basic_clean(s)
        if not t:
            return None, []
        # 1) set { … }
        if t.startswith("{") and t.endswith("}"):
            return "set", self._split_top_level_commas(t[1:-1])
        # 2) interval or union of intervals: 각 조각이 ‘정확히 인터벌’일 때만
        def _is_interval_piece(piece: str) -> bool:
            return bool(self.INTERVAL_RX.fullmatch(piece.strip()))
        if "\\cup" in t:
            parts = re.split(r"\s*\\cup\s*", t)
            if parts and all(_is_interval_piece(p) for p in parts):
                return "interval", [t]
        else:
            if _is_interval_piece(t):
                return "interval", [t]
        # if (t.startswith("[") or t.startswith("(")) and (t.endswith("]") or t.endswith(")")):
        #     return "interval", [t]
        # 3) tuple: 최상위 콤마가 1개 이상일 때만
        if t.startswith("(") and t.endswith(")"):
            items = self._split_top_level_commas(t[1:-1])
            if len(items) >= 2:
                return "tuple", items
        # if t.startswith("(") and t.endswith(")"):
        #     return "tuple", self._split_top_level_commas(t[1:-1])
        # 4) bare CSV → tuple
        csv = self._csv_to_tuple_str(t)
        if csv:
            return "tuple", self._split_top_level_commas(csv[1:-1])
        return None, []

    def _compare_sets(self, pred_items: List[str], gold_items: List[str]) -> bool:
        used = [False] * len(gold_items)
        for p in pred_items:
            hit = False
            for j, g in enumerate(gold_items):
                if not used[j] and self.grade(p, g):
                    used[j] = True; hit = True; break
            if not hit:
                return False
        return all(used) and len(pred_items) == len(gold_items)

    def _compare_tuples(self, pred_items: List[str], gold_items: List[str]) -> bool:
        if len(pred_items) != len(gold_items):
            return False
        return all(self.grade(p, g) for p, g in zip(pred_items, gold_items))

    # tolerant containers (space/braket/infinity normalization)
    def _normalize_tuple_like(self, s: str) -> Optional[str]:
        if not s:
            return None
        t = s.strip()
        if t.startswith("(") and t.endswith(")"):
            # inner = t[1:-1].strip()
            # inner = self._SPACE_AFTER_COMMA.sub(",", inner)
            # inner = self._MULTI_SPACE.sub(" ", inner)
            # return f"({inner})"
            items = self._split_top_level_commas(t[1:-1])
            if len(items) >= 2:  # 콤마가 있어야 ‘튜플’로 인정
                inner = ",".join(x.strip() for x in items)
                inner = self._MULTI_SPACE.sub(" ", inner)
                return f"({inner})"
        return None

    def _normalize_interval_like(self, s: str) -> Optional[str]:
        if not s:
            return None
        t = self._balance_brackets(s.strip())
        # Fix broken '-\frac{ }{1}2' → '-1/2'
        t = re.sub(r"-?\\frac\s*\{\s*\}\s*\{\s*1\s*\}\s*2", "-1/2", t)
        # bare "0, \infty" → "[0, \infty)"
        if re.fullmatch(r"\s*0\s*,\s*[^)\]]+\s*\)?\]?\s*", t):
            right = t.split(",", 1)[1].strip()
            right = self._BARE_INF.sub(lambda m: r"\\infty", right)
            right = right.rstrip(")]")
            return f"[0,{right})"
        # Split on top-level \cup and normalize pieces
        parts = re.split(r"\s*\\cup\s*", t)
        ivs: List[str] = []
        for p in parts:
            p = p.strip()
            m = re.search(r"([\[(])\s*([^,]+)\s*,\s*([^\])\)]+)\s*([\])])", p)
            if not m:
                # more flexible check
                m2 = re.search(r"^\s*(?P<L>[\[(])?\s*(?P<a>[^,]+)\s*,\s*(?P<b>[^\])\)]+)\s*(?P<R>[\])])?\s*$", p)
                if not m2:
                    return None
                L = m2.group("L") or "("
                R = m2.group("R") or ")"
                a = m2.group("a"); b = m2.group("b")
            else:
                L, a, b, R = m.groups()
            a = re.sub(r"\s+", "", a.strip())
            b = re.sub(r"\s+", "", b.strip())
            a = self._BARE_INF.sub(lambda m: "-\\infty" if a.lstrip().startswith("-") else "\\infty", a)
            b = self._BARE_INF.sub(lambda m: "\\infty", b)
            ivs.append(f"{L}{a},{b}{R}")
        return r" \cup ".join(ivs) if ivs else None

    def _containers_equivalent(self, p: str, g: str) -> Optional[bool]:
        pt = self._normalize_tuple_like(p); gt = self._normalize_tuple_like(g)
        if pt and gt:
            if pt == gt:
                return True
            # 다르면 확정 False 내리지 말고 포기(None)해서 아래 구조적 비교로 넘어가게 함
            return None
        pi = self._normalize_interval_like(p); gi = self._normalize_interval_like(g)
        if pi and gi:
            if pi == gi:
                return True
            return None
        return None

    # ---------------------------- parsing nums --------------------------
    def _parse_numeric(self, s: str):
        if s is None:
            return None
        t = str(s).strip()
        if not t:
            return None
        if t in {self.NO_SOLUTION, self.DNE, self.INF}:
            return t
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

    # ----------------------------- matrix compare ------------------------
    def _parse_matrix(self, s: str) -> Optional[List[List[str]]]:
        m = self._MAT_RX.search(s)
        if not m:
            return None
        body = m.group(1)
        # 행: '\\\\', 열: '&'
        rows = [r.strip() for r in re.split(r"\\\\", body) if r.strip()]
        mat: List[List[str]] = []
        for r in rows:
            cols = [c.strip() for c in r.split("&")]
            mat.append(cols)
        return mat

    def _cell_equal(self, a: str, b: str) -> bool:
        # 최소 비용 비교: 문자열 정규화 → 숫자 → 심볼릭
        pa = self._normalize_atomic(a); pb = self._normalize_atomic(b)
        if pa == pb and pa != "":
            return True
        na = self._parse_numeric(pa); nb = self._parse_numeric(pb)
        if na is not None and nb is not None:
            if isinstance(na, Fraction) and isinstance(nb, Fraction):
                return na == nb
            try:
                va = float(na) if not isinstance(na, float) else na
                vb = float(nb) if not isinstance(nb, float) else nb
                return math.isfinite(va) and math.isfinite(vb) and abs(va - vb) <= self.atol
            except Exception:
                pass
        return self._sympy_equal(pa, pb)
    
    def _maybe_compare_matrix(self, p: str, g: str) -> Optional[bool]:
        mp = self._parse_matrix(p)
        mg = self._parse_matrix(g)
        if mp is None or mg is None:
            return None
        if len(mp) != len(mg) or any(len(r1) != len(r2) for r1, r2 in zip(mp, mg)):
            return False
        for r1, r2 in zip(mp, mg):
            for a, b in zip(r1, r2):
                if not self._cell_equal(a, b):
                    return False
        return True
    
    # ----------------------------- public: grade ------------------------
    def grade(self, pred: str, gold: str) -> bool:
        """
        Equivalence check for MATH/GSM-style answers.
         0) Special tokens short-circuit.
         1) Containers: set/tuple/interval exact (structure) or tolerant (spacing/brackets/union).
         2) Scalars: normalized string ==, numeric == (fractions/decimals), symbolic equality (sympy),
            with ± expansion.
        """
        if pred is None or gold is None:
            return False
        p0 = self._normalize_special_tokens(pred)
        g0 = self._normalize_special_tokens(gold)

        # explicit special tokens
        if p0 in {self.NO_SOLUTION, self.DNE, self.INF} or g0 in {self.NO_SOLUTION, self.DNE, self.INF}:
            return p0 == g0
        
        # matrix comparison
        mat_eq = self._maybe_compare_matrix(p0, g0)
        if mat_eq is not None:
            return bool(mat_eq)
        
        # tolerant container equivalence 
        tol = self._containers_equivalent(p0, g0)
        if tol is True:
            return True

        # container dispatch (no set↔tuple mixing)
        pk, pitems = self._as_container(p0)
        gk, gitems = self._as_container(g0)
        if pk and gk:
            if pk != gk:
                return False
            if pk == "set":
                return self._compare_sets(pitems, gitems)
            if pk == "tuple":
                return self._compare_tuples(pitems, gitems)
            if pk == "interval":
                return self._sympy_equal(p0, g0)

        # scalars with ± branching
        for pc in self._expand_pm(p0):
            pn = self._normalize_atomic(pc)
            for gc in self._expand_pm(g0):
                gn = self._normalize_atomic(gc)
                if pn == gn and pn != "":
                    return True
                p_num = self._parse_numeric(pn)
                g_num = self._parse_numeric(gn)
                if p_num is not None and g_num is not None:
                    if isinstance(p_num, Fraction) and isinstance(g_num, Fraction):
                        return p_num == g_num
                    try:
                        pv = float(p_num) if not isinstance(p_num, float) else p_num
                        gv = float(g_num) if not isinstance(g_num, float) else g_num
                        if math.isfinite(pv) and math.isfinite(gv) and abs(pv - gv) <= self.atol:
                            return True
                    except Exception:
                        pass
                # symbolic equality 
                if self._sympy_equal(pn, gn) or self._sympy_equal(p0, g0):
                    return True
        return False

# --------------------------- module-level API ---------------------------
MATHScorer = MathScorer(atol=1e-6)

# # Test # 
# if __name__ == "__main__":
#     scorer = MathScorer(atol=1e-6)
#     pred = scorer.extract_pred
#     gold = scorer.extract_gold
#     grader = scorer.grade

#     tests = [
#         ("We have $a=-\\frac 12$ and $b=\\boxed{\\frac 54}$.", "[\\boxed{\\frac{5}{4}}.\\]", True),
#         ("&=\\boxed{\\frac{\\sqrt6}3}.\n\\end{align*}", "[\n\\boxed{\\frac{\\sqrt{6}}{3}}\n\\]", True),
#         ("Thus, the matrix is $\\boxed{\\begin{pmatrix} -4/5 & -3/5 \\\\ -3/5 & 4/5 \\end{pmatrix}}.$", "is:\n\\[\n\\boxed{\\begin{pmatrix} -\\frac{4}{5} & -\\frac{3}{5} \\\\ -\\frac{3}{5} & \\frac{4}{5} \\end{pmatrix}}\n\\]", True),
#         ("90 square units", "90", True),
#         ("\\frac{1}{33}", "\\dfrac{1}{33}", True),
#         ("\\boxed{5\\sqrt{2}}", "sqrt{50}", True),
#         ("-\\infty, -7)\\cup(-7, 3)\\cup(3, \\infty", "-\\infty, -7) \\cup (-7, 3) \\cup (3, \\infty", True),
#         ("\\frac{68}{3} pounds", "[ \\boxed{\\frac{68}{3}} \\]", True),
#         ("$r^2 + 10r+25 = \\boxed{(r+5)^2}", " the factored form of \\( r^2 + 10r + 25 \\) is \\(\\boxed{(r + 5)^2}\\).", True),
#         ("\\frac{x + 2}{7}}", "x/7 + 2/7", True),
#         (".5", "(\\boxed{\\frac{1}{2}}\\)", True),
#         ("(4,6,14,15)", "(4, 6, 14, 15)", True),
#         ("$x+2$ have opposite signs, so $-2 \\le x \\le 7$ and $\\boxed{x \\in [-2,7]}$.", """The final answer is:\n\n\\[\\boxed{[-2, 7]}\\]""", True),
#         ("Note that $-16x^4+x^2+2x+1=(x+1)^2-(4x^2)^2=\\boxed{(-4x^2+x+1)(4x^2+x+1)}$, where we have used the difference of squares identity for the second equality.", "final answer is:\n\n\\[\n\\boxed{(-4x^2 + x + 1)(4x^2 + x + 1)}\n\\]", True),
#         ("the cylindrical coordinates of the point \\((1, -1, -6)\\) are:\n\n\\[ \\boxed{\\left(\\sqrt{2}, \\frac{7\\pi}{4}, -6\\right)} \\]", "so the cylindrical coordinates are $\\boxed{\\left( \\sqrt{2}, \\frac{7 \\pi}{4}, -6 \\right)}.$", True),
#         ("Since $\\cos \\frac{3 \\pi}{4} = -\\frac{1}{\\sqrt{2}},$ $\\arccos \\left( -\\frac{1}{\\sqrt{2}} \\right) = \\boxed{\\frac{3 \\pi}{4}}.", "answer is:\n\\[\n\\boxed{\\frac{3\\pi}{4}}\n\\]", True),
#     ]

#     for i, (p, g, want) in enumerate(tests, 1):
#         p = pred(p)
#         rec = {"solution": g}
#         g = gold(rec)
#         got = grader(p, g)
#         print(f"[{i}] {p!r} vs {g!r} -> {got} (want {want})")

