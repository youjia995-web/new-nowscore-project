import argparse
import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from openpyxl import load_workbook
import math

KEYS = {
    "date": ["日期", "date", "比赛日期", "matchdate", "date"],
    "league": ["联赛", "league", "unnamed_2"],
    "season": ["赛季", "season", "比赛信息"],
    "home_team": ["主队", "主队名称", "基本面_2", "hometeam", "home team", "home_team", "team name"],
    "away_team": ["客队", "客队名称", "基本面_9", "awayteam", "away team", "away_team", "away tema"],
    "home_goals": ["主队进球", "fthg", "全场主队进球", "半全场数据_6", "hg", "homegoals", "home_goals"],
    "away_goals": ["客队进球", "ftag", "全场客队进球", "半全场数据_7", "ag", "awaygoals", "away_goals"],
    "score": ["比分", "全场比分", "ft", "fulltime", "score"],
    "home_xg": ["主队xg", "主xg", "主队预期进球", "xgh", "home_xg"],
    "away_xg": ["客队xg", "客xg", "客队预期进球", "xga_match", "away_xg"],
    "home_rank": ["home team rank", "home_rank", "rank_home", "主队排名", "主排名"],
    "away_rank": ["away team rank", "away_rank", "rank_away", "客队排名", "客排名"],
    "round": ["round", "轮次", "match round", "轮", "比赛轮次"],
    "odds_home_open": ["open_home_odds", "home_odds_open"],
    "odds_draw_open": ["open_draw_odds", "draw_odds_open"],
    "odds_away_open": ["open_away_odds", "away_odds_open"],
    "odds_home_close": ["close_home_odds"],
    "odds_draw_close": ["close_draw_odds"],
    "odds_away_close": ["close_away_odds"],
    "ah_line_open": ["open_ah_line", "ah_open_line"],
    "ah_home_price_open": ["open_ah_home_price"],
    "ah_away_price_open": ["open_ah_away_price"],
    "ah_line_close": ["close_ah_line"],
    "ah_home_price_close": ["close_ah_home_price"],
    "ah_away_price_close": ["close_ah_away_price"],
}

def _norm(s: str) -> str:
    t = str(s or "").strip().lower()
    return "".join(t.split())

def _find_col(df_cols, keys):
    cols_norm = { _norm(c): c for c in df_cols }
    for k in keys:
        kn = _norm(k)
        if kn in cols_norm:
            return cols_norm[kn]
    return None

def _parse_score(s):
    if s is None:
        return None, None
    t = str(s).strip()
    import re
    m = re.match(r"^(\d+)\s*[-:：]\s*(\d+)$", t)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def _to_int(v):
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None

def _to_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None

class SeqDataset(Dataset):
    def __init__(self, hX, aX, y, mean, std):
        self.hX = torch.tensor(hX, dtype=torch.float32)
        self.aX = torch.tensor(aX, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.mean = mean
        self.std = std
    def __len__(self):
        return self.y.shape[0]
    def __getitem__(self, i):
        hs = (self.hX[i] - self.mean) / self.std
        as_ = (self.aX[i] - self.mean) / self.std
        return hs, as_, self.y[i]

class TwinGRU(nn.Module):
    def __init__(self, input_dim=5, hidden=64, dropout=0.4, num_classes=3):
        super().__init__()
        self.h_gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.a_gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes))
    def forward(self, h_seq, a_seq):
        _, h = self.h_gru(h_seq)
        _, a = self.a_gru(a_seq)
        x = torch.cat([h[-1], a[-1]], dim=1)
        return self.fc(x)

def load_matches_from_excel(path: str, sheet=None):
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    if isinstance(df, dict):
        df = next(iter(df.values()))
    cols = [str(c) for c in df.columns.tolist()]
    print("cols", cols[:30])
    if len(df) > 0:
        print("row0", {str(k): str(df.iloc[0][k]) for k in df.columns[:30]})
    col_map = {k: _find_col(cols, v) for k, v in KEYS.items()}
    if not any(col_map.values()) and len(df) > 0:
        row0_vals = [str(df.iloc[0][c]) if df.iloc[0][c] is not None else "" for c in df.columns]
        col_map2 = {k: _find_col(row0_vals, v) for k, v in KEYS.items()}
        if any(col_map2.values()):
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = row0_vals
            cols = [str(c) for c in df.columns.tolist()]
            col_map = {k: _find_col(cols, v) for k, v in KEYS.items()}
    print("col_map", col_map)
    rows = []
    for _, row in df.iterrows():
        ht = row[col_map.get("home_team")] if col_map.get("home_team") else None
        at = row[col_map.get("away_team")] if col_map.get("away_team") else None
        if ht is None or at is None:
            continue
        hg = row[col_map.get("home_goals")] if col_map.get("home_goals") else None
        ag = row[col_map.get("away_goals")] if col_map.get("away_goals") else None
        if (hg is None or (isinstance(hg, float) and pd.isna(hg))) and col_map.get("score"):
            h2, a2 = _parse_score(row[col_map.get("score")])
            hg = hg if hg is not None else h2
            ag = ag if ag is not None else a2
        hxg = row[col_map.get("home_xg")] if col_map.get("home_xg") else None
        axg = row[col_map.get("away_xg")] if col_map.get("away_xg") else None
        hrank = row[col_map.get("home_rank")] if col_map.get("home_rank") else None
        arank = row[col_map.get("away_rank")] if col_map.get("away_rank") else None
        rnd = row[col_map.get("round")] if col_map.get("round") else None
        oh_open = row[col_map.get("odds_home_open")] if col_map.get("odds_home_open") else None
        od_open = row[col_map.get("odds_draw_open")] if col_map.get("odds_draw_open") else None
        oa_open = row[col_map.get("odds_away_open")] if col_map.get("odds_away_open") else None
        oh_close = row[col_map.get("odds_home_close")] if col_map.get("odds_home_close") else None
        od_close = row[col_map.get("odds_draw_close")] if col_map.get("odds_draw_close") else None
        oa_close = row[col_map.get("odds_away_close")] if col_map.get("odds_away_close") else None
        ah_line_open = row[col_map.get("ah_line_open")] if col_map.get("ah_line_open") else None
        ah_hp_open = row[col_map.get("ah_home_price_open")] if col_map.get("ah_home_price_open") else None
        ah_ap_open = row[col_map.get("ah_away_price_open")] if col_map.get("ah_away_price_open") else None
        ah_line_close = row[col_map.get("ah_line_close")] if col_map.get("ah_line_close") else None
        ah_hp_close = row[col_map.get("ah_home_price_close")] if col_map.get("ah_home_price_close") else None
        ah_ap_close = row[col_map.get("ah_away_price_close")] if col_map.get("ah_away_price_close") else None
        date_val = row[col_map.get("date")] if col_map.get("date") else None
        d = None
        if isinstance(date_val, datetime):
            d = date_val
        elif isinstance(date_val, str):
            try:
                d = datetime.fromisoformat(date_val.strip())
            except Exception:
                d = None
        rows.append({
            "date": d,
            "season": str(row[col_map.get("season")]).strip() if col_map.get("season") else None,
            "home_team": str(ht).strip(),
            "away_team": str(at).strip(),
            "home_goals": _to_int(hg),
            "away_goals": _to_int(ag),
            "home_xg": _to_float(hxg),
            "away_xg": _to_float(axg),
            "home_rank": _to_float(hrank),
            "away_rank": _to_float(arank),
            "round": _to_int(rnd),
            "odds_home_open": _to_float(oh_open),
            "odds_draw_open": _to_float(od_open),
            "odds_away_open": _to_float(oa_open),
            "odds_home_close": _to_float(oh_close),
            "odds_draw_close": _to_float(od_close),
            "odds_away_close": _to_float(oa_close),
            "ah_line_open": _to_float(ah_line_open),
            "ah_home_price_open": _to_float(ah_hp_open),
            "ah_away_price_open": _to_float(ah_ap_open),
            "ah_line_close": _to_float(ah_line_close),
            "ah_home_price_close": _to_float(ah_hp_close),
            "ah_away_price_close": _to_float(ah_ap_close),
        })
    rows = [r for r in rows if r["home_goals"] is not None and r["away_goals"] is not None]
    rows.sort(key=lambda r: (r["date"] is None, r["date"] or datetime.min))
    print("rows", len(rows))
    return rows

def build_sequences(df, seq_len: int):
    hist = {}
    hX = []
    aX = []
    y = []
    meta = []
    feats = [
        "gf", "ga", "xgf", "xga", "is_home", "rank", "round",
        "odds_h_o", "odds_d_o", "odds_a_o",
        "odds_h_c", "odds_d_c", "odds_a_c",
        "ah_line_o", "ah_hp_o", "ah_ap_o",
        "ah_line_c",
        "prob_h_o", "prob_d_o", "prob_a_o",
        "prob_h_c", "prob_d_c", "prob_a_c",
        "delta_h", "delta_d", "delta_a",
        "over_o", "over_c",
        "log_ratio_o", "log_ratio_c",
        "ah_p_h_o", "ah_p_h_c"
    ]
    for r in df:
        ht = r["home_team"]
        at = r["away_team"]
        if ht not in hist:
            hist[ht] = []
        if at not in hist:
            hist[at] = []
        if len(hist[ht]) >= seq_len and len(hist[at]) >= seq_len:
            h_seq = hist[ht][-seq_len:]
            a_seq = hist[at][-seq_len:]
            label = 0 if r["home_goals"] > r["away_goals"] else (2 if r["home_goals"] < r["away_goals"] else 1)
            hX.append(h_seq)
            aX.append(a_seq)
            y.append(label)
            meta.append({"date": r.get("date"), "season": r.get("season")})
        h_rec = {
            "gf": float(r["home_goals"] or 0),
            "ga": float(r["away_goals"] or 0),
            "xgf": float((r["home_xg"] if r["home_xg"] is not None else 0.0)),
            "xga": float((r["away_xg"] if r["away_xg"] is not None else 0.0)),
            "is_home": 1.0,
            "rank": float(r["home_rank"] if r.get("home_rank") is not None else 0.0),
            "round": float(r["round"] if r.get("round") is not None else 0.0),
            "odds_h_o": float(r["odds_home_open"] if r.get("odds_home_open") is not None else 0.0),
            "odds_d_o": float(r["odds_draw_open"] if r.get("odds_draw_open") is not None else 0.0),
            "odds_a_o": float(r["odds_away_open"] if r.get("odds_away_open") is not None else 0.0),
            "odds_h_c": float(r["odds_home_close"] if r.get("odds_home_close") is not None else 0.0),
            "odds_d_c": float(r["odds_draw_close"] if r.get("odds_draw_close") is not None else 0.0),
            "odds_a_c": float(r["odds_away_close"] if r.get("odds_away_close") is not None else 0.0),
            "ah_line_o": float(r["ah_line_open"] if r.get("ah_line_open") is not None else 0.0),
            "ah_hp_o": float(r["ah_home_price_open"] if r.get("ah_home_price_open") is not None else 0.0),
            "ah_ap_o": float(r["ah_away_price_open"] if r.get("ah_away_price_open") is not None else 0.0),
            "ah_line_c": float(r["ah_line_close"] if r.get("ah_line_close") is not None else 0.0),
        }
        a_rec = {
            "gf": float(r["away_goals"] or 0),
            "ga": float(r["home_goals"] or 0),
            "xgf": float((r["away_xg"] if r["away_xg"] is not None else 0.0)),
            "xga": float((r["home_xg"] if r["home_xg"] is not None else 0.0)),
            "is_home": 0.0,
            "rank": float(r["away_rank"] if r.get("away_rank") is not None else 0.0),
            "round": float(r["round"] if r.get("round") is not None else 0.0),
            "odds_h_o": float(r["odds_home_open"] if r.get("odds_home_open") is not None else 0.0),
            "odds_d_o": float(r["odds_draw_open"] if r.get("odds_draw_open") is not None else 0.0),
            "odds_a_o": float(r["odds_away_open"] if r.get("odds_away_open") is not None else 0.0),
            "odds_h_c": float(r["odds_home_close"] if r.get("odds_home_close") is not None else 0.0),
            "odds_d_c": float(r["odds_draw_close"] if r.get("odds_draw_close") is not None else 0.0),
            "odds_a_c": float(r["odds_away_close"] if r.get("odds_away_close") is not None else 0.0),
            "ah_line_o": float(r["ah_line_open"] if r.get("ah_line_open") is not None else 0.0),
            "ah_hp_o": float(r["ah_home_price_open"] if r.get("ah_home_price_open") is not None else 0.0),
            "ah_ap_o": float(r["ah_away_price_open"] if r.get("ah_away_price_open") is not None else 0.0),
            "ah_line_c": float(r["ah_line_close"] if r.get("ah_line_close") is not None else 0.0),
        }
        p_h_o = (1.0 / h_rec["odds_h_o"]) if h_rec["odds_h_o"] > 0 else 0.0
        p_d_o = (1.0 / h_rec["odds_d_o"]) if h_rec["odds_d_o"] > 0 else 0.0
        p_a_o = (1.0 / h_rec["odds_a_o"]) if h_rec["odds_a_o"] > 0 else 0.0
        s_o = p_h_o + p_d_o + p_a_o
        p_h_o_n = (p_h_o / s_o) if s_o > 0 else 0.0
        p_d_o_n = (p_d_o / s_o) if s_o > 0 else 0.0
        p_a_o_n = (p_a_o / s_o) if s_o > 0 else 0.0
        over_o = (p_h_o + p_d_o + p_a_o) - 1.0
        p_h_c = (1.0 / h_rec["odds_h_c"]) if h_rec["odds_h_c"] > 0 else 0.0
        p_d_c = (1.0 / h_rec["odds_d_c"]) if h_rec["odds_d_c"] > 0 else 0.0
        p_a_c = (1.0 / h_rec["odds_a_c"]) if h_rec["odds_a_c"] > 0 else 0.0
        s_c = p_h_c + p_d_c + p_a_c
        p_h_c_n = (p_h_c / s_c) if s_c > 0 else 0.0
        p_d_c_n = (p_d_c / s_c) if s_c > 0 else 0.0
        p_a_c_n = (p_a_c / s_c) if s_c > 0 else 0.0
        over_c = (p_h_c + p_d_c + p_a_c) - 1.0
        delta_h = h_rec["odds_h_c"] - h_rec["odds_h_o"]
        delta_d = h_rec["odds_d_c"] - h_rec["odds_d_o"]
        delta_a = h_rec["odds_a_c"] - h_rec["odds_a_o"]
        log_ratio_o = math.log(h_rec["odds_a_o"] / h_rec["odds_h_o"]) if h_rec["odds_a_o"] > 0 and h_rec["odds_h_o"] > 0 else 0.0
        log_ratio_c = math.log(h_rec["odds_a_c"] / h_rec["odds_h_c"]) if h_rec["odds_a_c"] > 0 and h_rec["odds_h_c"] > 0 else 0.0
        h_rec.update({
            "prob_h_o": p_h_o_n,
            "prob_d_o": p_d_o_n,
            "prob_a_o": p_a_o_n,
            "prob_h_c": p_h_c_n,
            "prob_d_c": p_d_c_n,
            "prob_a_c": p_a_c_n,
            "delta_h": delta_h,
            "delta_d": delta_d,
            "delta_a": delta_a,
            "over_o": over_o,
            "over_c": over_c,
            "log_ratio_o": log_ratio_o,
            "log_ratio_c": log_ratio_c,
            "ah_p_h_o": h_rec["ah_line_o"] * p_h_o_n,
            "ah_p_h_c": h_rec["ah_line_c"] * p_h_c_n,
        })
        a_rec.update({
            "prob_h_o": p_h_o_n,
            "prob_d_o": p_d_o_n,
            "prob_a_o": p_a_o_n,
            "prob_h_c": p_h_c_n,
            "prob_d_c": p_d_c_n,
            "prob_a_c": p_a_c_n,
            "delta_h": delta_h,
            "delta_d": delta_d,
            "delta_a": delta_a,
            "over_o": over_o,
            "over_c": over_c,
            "log_ratio_o": log_ratio_o,
            "log_ratio_c": log_ratio_c,
            "ah_p_h_o": h_rec["ah_line_o"] * p_h_o_n,
            "ah_p_h_c": h_rec["ah_line_c"] * p_h_c_n,
        })
        hist[ht].append([h_rec[f] for f in feats])
        hist[at].append([a_rec[f] for f in feats])
    return hX, aX, y, meta

def eval_metrics(model, dl, device):
    model.eval()
    n = 0
    correct = 0
    logloss_sum = 0.0
    brier_sum = 0.0
    with torch.no_grad():
        for h_seq, a_seq, y in dl:
            h_seq = h_seq.to(device)
            a_seq = a_seq.to(device)
            y = y.to(device)
            logits = model(h_seq, a_seq)
            prob = torch.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)
            correct += (pred == y).sum().item()
            p_true = prob.gather(1, y.unsqueeze(1)).squeeze(1).clamp(min=1e-9)
            logloss_sum += float((-torch.log(p_true)).sum().item())
            oh = torch.nn.functional.one_hot(y, num_classes=3).float()
            brier = ((prob - oh) ** 2).sum(dim=1).sum().item()
            brier_sum += float(brier)
            n += y.shape[0]
    acc = correct / max(n, 1)
    logloss = logloss_sum / max(n, 1)
    brier = brier_sum / max(n, 1)
    return acc, logloss, brier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("excel_path")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--cv-folds", type=int, default=1)
    ap.add_argument("--cv-by-season", action="store_true")
    ap.add_argument("--split-by-season", action="store_true")
    ap.add_argument("--val-seasons", type=int, default=1)
    args = ap.parse_args()

    df = load_matches_from_excel(args.excel_path, args.sheet)
    hX, aX, y, meta = build_sequences(df, args.seq_len)
    if len(y) == 0:
        print("no_samples")
        return
    k = int(len(y) * (1.0 - max(0.05, min(0.5, args.val_ratio))))
    if args.cv_by_season and args.cv_folds and args.cv_folds > 1:
        seasons = []
        seen = set()
        for m in meta:
            s = m.get("season")
            if s not in seen and s is not None:
                seen.add(s)
                seasons.append(s)
        if len(seasons) < 2:
            print("no_samples")
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_all = None
        best_state_all = None
        acc_sum = 0.0
        ll_sum = 0.0
        br_sum = 0.0
        used_folds = 0
        for i, s in enumerate(seasons[1:], start=1):
            idx_va = [ix for ix in range(len(y)) if meta[ix].get("season") == s]
            idx_tr = [ix for ix in range(len(y)) if meta[ix].get("season") in seasons[:i]]
            if len(idx_tr) == 0 or len(idx_va) == 0:
                continue
            hX_tr = [hX[ix] for ix in idx_tr]
            aX_tr = [aX[ix] for ix in idx_tr]
            y_tr = [y[ix] for ix in idx_tr]
            hX_va = [hX[ix] for ix in idx_va]
            aX_va = [aX[ix] for ix in idx_va]
            y_va = [y[ix] for ix in idx_va]
            ft = torch.tensor(hX_tr + aX_tr, dtype=torch.float32).reshape(-1, len(hX_tr[0][0]))
            mean = ft.mean(dim=0)
            std = ft.std(dim=0)
            std = torch.where(std == 0, torch.ones_like(std), std)
            ds_tr = SeqDataset(hX_tr, aX_tr, y_tr, mean, std)
            ds_va = SeqDataset(hX_va, aX_va, y_va, mean, std)
            dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True)
            dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False)
            model = TwinGRU(input_dim=len(hX_tr[0][0]), hidden=args.hidden, dropout=args.dropout).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
            y_tr_tensor = torch.tensor(y_tr, dtype=torch.long)
            class_counts = torch.bincount(y_tr_tensor, minlength=3).float()
            weights = class_counts.sum() / (class_counts + 1e-6)
            weights = weights / weights.sum() * 3.0
            crit = nn.CrossEntropyLoss(weight=weights.to(device))
            best = None
            best_state = None
            for ep in range(args.epochs):
                model.train()
                for h_seq, a_seq, yb in dl_tr:
                    h_seq = h_seq.to(device)
                    a_seq = a_seq.to(device)
                    yb = yb.to(device)
                    opt.zero_grad()
                    logits = model(h_seq, a_seq)
                    loss = crit(logits, yb)
                    loss.backward()
                    opt.step()
                acc, ll, br = eval_metrics(model, dl_va, device)
                score = -ll
                if (best is None) or (score > best):
                    best = score
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                print(f"fold_season={s} epoch={ep+1} val_acc={acc:.4f} val_logloss={ll:.4f} val_brier={br:.4f}")
            if best_state is not None:
                model.load_state_dict(best_state)
            acc, ll, br = eval_metrics(model, dl_va, device)
            acc_sum += acc
            ll_sum += ll
            br_sum += br
            used_folds += 1
            if (best_all is None) or (-ll > best_all):
                best_all = -ll
                best_state_all = {k: v.cpu() for k, v in model.state_dict().items()}
        if used_folds > 0:
            acc_mean = acc_sum / used_folds
            ll_mean = ll_sum / used_folds
            br_mean = br_sum / used_folds
            base = Path(__file__).resolve().parents[2]
            out = base / "lstm_gru.pth"
            if best_state_all is not None:
                model = TwinGRU(input_dim=len(hX[0][0]), hidden=args.hidden, dropout=args.dropout)
                model.load_state_dict(best_state_all)
                torch.save(model.state_dict(), str(out))
            print(f"cv_mean_val_acc={acc_mean:.4f} cv_mean_val_logloss={ll_mean:.4f} cv_mean_val_brier={br_mean:.4f}")
            print("saved", str(out))
            return
        N = len(y)
        fold_size = max(1, N // args.cv_folds)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_all = None
        best_state_all = None
        acc_sum = 0.0
        ll_sum = 0.0
        br_sum = 0.0
        used_folds = 0
        for i in range(1, args.cv_folds):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if (i + 1) * fold_size <= N else N
            hX_tr, aX_tr, y_tr = hX[:val_start], aX[:val_start], y[:val_start]
            hX_va, aX_va, y_va = hX[val_start:val_end], aX[val_start:val_end], y[val_start:val_end]
            if len(y_tr) == 0 or len(y_va) == 0:
                continue
            ft = torch.tensor(hX_tr + aX_tr, dtype=torch.float32).reshape(-1, len(hX_tr[0][0]))
            mean = ft.mean(dim=0)
            std = ft.std(dim=0)
            std = torch.where(std == 0, torch.ones_like(std), std)
            ds_tr = SeqDataset(hX_tr, aX_tr, y_tr, mean, std)
            ds_va = SeqDataset(hX_va, aX_va, y_va, mean, std)
            dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True)
            dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False)
            model = TwinGRU(input_dim=len(hX_tr[0][0]), hidden=args.hidden, dropout=args.dropout).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
            y_tr_tensor = torch.tensor(y_tr, dtype=torch.long)
            class_counts = torch.bincount(y_tr_tensor, minlength=3).float()
            weights = class_counts.sum() / (class_counts + 1e-6)
            weights = weights / weights.sum() * 3.0
            crit = nn.CrossEntropyLoss(weight=weights.to(device))
            best = None
            best_state = None
            for ep in range(args.epochs):
                model.train()
                for h_seq, a_seq, yb in dl_tr:
                    h_seq = h_seq.to(device)
                    a_seq = a_seq.to(device)
                    yb = yb.to(device)
                    opt.zero_grad()
                    logits = model(h_seq, a_seq)
                    loss = crit(logits, yb)
                    loss.backward()
                    opt.step()
                acc, ll, br = eval_metrics(model, dl_va, device)
                score = -ll
                if (best is None) or (score > best):
                    best = score
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                print(f"fold={i} epoch={ep+1} val_acc={acc:.4f} val_logloss={ll:.4f} val_brier={br:.4f}")
            if best_state is not None:
                model.load_state_dict(best_state)
            acc, ll, br = eval_metrics(model, dl_va, device)
            acc_sum += acc
            ll_sum += ll
            br_sum += br
            used_folds += 1
            if (best_all is None) or (-ll > best_all):
                best_all = -ll
                best_state_all = {k: v.cpu() for k, v in model.state_dict().items()}
        if used_folds > 0:
            acc_mean = acc_sum / used_folds
            ll_mean = ll_sum / used_folds
            br_mean = br_sum / used_folds
            base = Path(__file__).resolve().parents[2]
            out = base / "lstm_gru.pth"
            if best_state_all is not None:
                model = TwinGRU(input_dim=len(hX[0][0]), hidden=args.hidden, dropout=args.dropout)
                model.load_state_dict(best_state_all)
                torch.save(model.state_dict(), str(out))
            print(f"cv_mean_val_acc={acc_mean:.4f} cv_mean_val_logloss={ll_mean:.4f} cv_mean_val_brier={br_mean:.4f}")
            print("saved", str(out))
        else:
            print("no_samples")
    else:
        hX_tr, aX_tr, y_tr = hX[:k], aX[:k], y[:k]
        hX_va, aX_va, y_va = hX[k:], aX[k:], y[k:]
        ft = torch.tensor(hX_tr + aX_tr, dtype=torch.float32).reshape(-1, len(hX_tr[0][0]))
        mean = ft.mean(dim=0)
        std = ft.std(dim=0)
        std = torch.where(std == 0, torch.ones_like(std), std)
        ds_tr = SeqDataset(hX_tr, aX_tr, y_tr, mean, std)
        ds_va = SeqDataset(hX_va, aX_va, y_va, mean, std)
        dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True)
        dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TwinGRU(input_dim=len(hX_tr[0][0]), hidden=args.hidden, dropout=args.dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        y_tr_tensor = torch.tensor(y_tr, dtype=torch.long)
        class_counts = torch.bincount(y_tr_tensor, minlength=3).float()
        weights = class_counts.sum() / (class_counts + 1e-6)
        weights = weights / weights.sum() * 3.0
        crit = nn.CrossEntropyLoss(weight=weights.to(device))
        best = None
        best_state = None
        for ep in range(args.epochs):
            model.train()
            for h_seq, a_seq, yb in dl_tr:
                h_seq = h_seq.to(device)
                a_seq = a_seq.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                logits = model(h_seq, a_seq)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
            acc, ll, br = eval_metrics(model, dl_va, device)
            score = -ll
            if (best is None) or (score > best):
                best = score
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"epoch={ep+1} val_acc={acc:.4f} val_logloss={ll:.4f} val_brier={br:.4f}")
        if best_state is not None:
            model.load_state_dict(best_state)
        acc, ll, br = eval_metrics(model, dl_va, device)
        base = Path(__file__).resolve().parents[2]
        out = base / "lstm_gru.pth"
        torch.save(model.state_dict(), str(out))
        print("saved", str(out))
        print(f"final_val_acc={acc:.4f} final_val_logloss={ll:.4f} final_val_brier={br:.4f}")

if __name__ == "__main__":
    main()