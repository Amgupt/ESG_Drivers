"""Robustness analysis for the ranking of ESG adoption drivers.

Reads the sheets 'Coding' and 'Included studies' from the data workbook and reproduces
every table in the sheet 'Robustness'.

Usage:  python robustness_analysis.py ESG_Adoption_Drivers_Data.xlsx
Requires: openpyxl, scipy
"""
import sys, statistics as st
from collections import defaultdict, Counter
import openpyxl
from scipy.stats import spearmanr, rankdata

CODES = [f'F{i}' for i in range(1, 21)] + ['Formal Institutions', 'Informal Institutions', 'Digital transformation']

def load(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    c, t = wb['Coding'], wb['Included studies']
    rows = [(r[0], r[3], r[7]) for r in c.iter_rows(min_row=2, values_only=True) if r[0]]   # paper, importance, code
    types = {f'Paper {t.cell(r, 1).value}': t.cell(r, 6).value for r in range(2, t.max_row + 1) if t.cell(r, 1).value}
    return rows, types

def scores(rows, papers, rule='MAX', norm='MAX'):
    """{driver: (frequency, normalised frequency, mean importance)} for a set of papers.
    rule: one score per paper and driver (MAX or MEAN of that paper's ratings).
    norm: MAX = frequency / highest frequency x 5; N = frequency / number of papers x 5; MINMAX = both measures scaled 0-1."""
    per = defaultdict(lambda: defaultdict(list)); coded = defaultdict(set)
    for p, sc, fc in rows:
        if p in papers and fc in CODES:
            coded[fc].add(p)
            if sc not in (None, ''): per[fc][p].append(float(sc))
    freq = {k: len(coded[k]) for k in CODES if coded[k] and per[k]}
    fmax, fmin = max(freq.values()), min(freq.values())
    mean = {k: st.mean((max(v) if rule == 'MAX' else st.mean(v)) for v in per[k].values()) for k in freq}
    mmax, mmin = max(mean.values()), min(mean.values())
    out = {}
    for k in freq:
        if norm == 'MAX':    nf, mi = freq[k] / fmax * 5, mean[k]
        elif norm == 'N':    nf, mi = freq[k] / len(papers) * 5, mean[k]
        elif norm == 'MINMAX': nf, mi = (freq[k] - fmin) / (fmax - fmin), (mean[k] - mmin) / (mmax - mmin)
        out[k] = (freq[k], nf, mi)
    return out

def composite(res, wf): return {k: wf * v[1] + (1 - wf) * v[2] for k, v in res.items()}
def rank_min(cw):
    vals = sorted(cw.values(), reverse=True); return {k: 1 + vals.index(v) for k, v in cw.items()}
def rho(a, b):
    ks = sorted(set(a) & set(b)); return spearmanr([a[k] for k in ks], [b[k] for k in ks]).correlation
def kendall_w(score_dicts):
    ks = sorted(set.intersection(*[set(r) for r in score_dicts])); m, n = len(score_dicts), len(ks)
    R = [rankdata([-r[k] for k in ks]) for r in score_dicts]
    tot = [sum(R[j][i] for j in range(m)) for i in range(n)]; mean = sum(tot) / n
    S = sum((x - mean) ** 2 for x in tot)
    T = sum(sum(t ** 3 - t for t in Counter(r).values()) for r in R)
    return 12 * S / (m ** 2 * (n ** 3 - n) - m * T)

def run(path):
    rows, types = load(path)
    allp = sorted({p for p, _, _ in rows}, key=lambda s: int(s.split()[1]))
    main = scores(rows, set(allp))
    SC = [('90/10', 0.9), ('60/40', 0.6), ('50/50', 0.5), ('40/60', 0.4), ('10/90', 0.1)]
    cws = {lab: composite(main, w) for lab, w in SC}; base = cws['50/50']
    out = {'drivers': sorted(main, key=lambda k: -base[k]), 'main': main,
           'scenario_ranks': {lab: rank_min(cw) for lab, cw in cws.items()},
           'scenario_rho': {lab: rho(cw, base) for lab, cw in cws.items()},
           'kendall_w': kendall_w(list(cws.values())),
           'sweep': {w / 10: rank_min(composite(main, w / 10)) for w in range(0, 11)}}
    out['sweep_rho'] = {w: rho(composite(main, w), base) for w in out['sweep']}
    reviews = {p for p, tp in types.items() if tp in ('LR', 'SLR')}
    alts = [('Main: MAX score, frequency / highest frequency', main),
            ('One score per paper = MEAN', scores(rows, set(allp), rule='MEAN')),
            ('Frequency / N', scores(rows, set(allp), norm='N')),
            ('Min-max scaling of both measures', scores(rows, set(allp), norm='MINMAX')),
            ('Without Paper 1 (source of the CSF taxonomy)', scores(rows, set(allp) - {'Paper 1'})),
            (f'Without the {len(reviews)} review articles', scores(rows, set(allp) - reviews))]
    out['alts'] = [(lab, rank_min(composite(r, 0.5)), rho(composite(r, 0.5), base), r) for lab, r in alts]
    out['reviews'] = sorted(reviews, key=lambda s: int(s.split()[1]))
    top5 = set(sorted(base, key=lambda k: -base[k])[:5]); loo = []; per_driver = defaultdict(list)
    for p in allp:
        cw = composite(scores(rows, set(allp) - {p}), 0.5); rk = rank_min(cw)
        t5 = set(sorted(cw, key=lambda k: -cw[k])[:5])
        loo.append((p, rho(cw, base), sorted(t5 - top5), sorted(top5 - t5)))
        for k in main: per_driver[k].append(rk.get(k))
    out['loo'] = loo
    out['loo_driver'] = {k: (min(x for x in v if x), max(x for x in v if x)) for k, v in per_driver.items()}
    return out

if __name__ == '__main__':
    o = run(sys.argv[1])
    print("Kendall's W (five scenarios):", round(o['kendall_w'], 3))
    for lab, rk, r, _ in o['alts']:
        print(f"{lab:48s} rho = {r:.3f}   top five: {[k for k, _ in sorted(rk.items(), key=lambda x: x[1])[:5]]}")
    print('Leave-one-study-out, lowest rho:', round(min(x[1] for x in o['loo']), 3))
