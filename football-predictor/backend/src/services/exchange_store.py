import io
import os
from typing import Dict, Optional, Any, Tuple

import pandas as pd


def _col(df: pd.DataFrame, candidates) -> Optional[str]:
    cols = [str(c).strip().lower() for c in df.columns]
    for c in candidates:
        c_lower = str(c).strip().lower()
        if c_lower in cols:
            return c_lower
    return None


def _as_float(x, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return default
        # 允许形如 "1.95" 或 "1.95(买盘)" 之类
        import re
        m = re.search(r"[+-]?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else default
    except Exception:
        return default


def _normalize_selection(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in {"", "nan", "none", "null"}:
        return None
    # 常见中文/英文别名
    mapping = {
        "主": "home",
        "主胜": "home",
        "主队": "home",
        "home": "home",
        "h": "home",
        "平": "draw",
        "和": "draw",
        "平局": "draw",
        "draw": "draw",
        "d": "draw",
        "客": "away",
        "客胜": "away",
        "客队": "away",
        "away": "away",
        "a": "away",
    }
    # 先严格匹配（避免单字母在 "nan" 等字符串中误匹配）
    if s in mapping:
        return mapping[s]
    # 再进行子串匹配，但排除单字母键
    for k in [kk for kk in mapping.keys() if len(kk) > 1]:
        if k in s:
            return mapping[k]
    return None


def _read_excel_from_bytes(buf: bytes, sheet: Optional[Any] = None) -> pd.DataFrame:
    bio = io.BytesIO(buf)
    sheet_name = 0 if sheet is None else sheet
    try:
        df = pd.read_excel(bio, sheet_name=sheet_name, engine="openpyxl")
    except Exception:
        df = pd.read_excel(bio, sheet_name=sheet_name)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _extract_long_format(df: pd.DataFrame) -> Tuple[Dict[str, float], float]:
    """
    长表结构：每行一个投注方向（主/平/客），含买盘/卖盘价格与成交量。
    返回：三路概率（未归一化）与总成交量估计。
    """
    sel_col = _col(df, [
        "selection", "投注项", "方向", "选项", "玩法", "胜平负", "主客", "结果", "outcome", "side", "项"
    ])
    if not sel_col:
        return {}, 0.0

    back_cols = [
        "back_odds", "best_back", "back price", "买盘价", "买入价", "买盘", "最高买盘", "价位"
    ]
    lay_cols = [
        "lay_odds", "best_lay", "lay price", "卖盘价", "卖出价", "卖盘", "最低卖盘", "价位"
    ]
    tv_cols = [
        "traded_volume", "matched", "成交量", "成交", "成交额", "成交金额"
    ]
    bv_cols = [
        "back_volume", "买盘量", "买入量", "挂买量", "挂单买", "买家挂牌"
    ]
    lv_cols = [
        "lay_volume", "卖盘量", "卖出量", "挂卖量", "挂单卖", "卖家挂牌"
    ]

    back_col = _col(df, back_cols)
    lay_col = _col(df, lay_cols)
    tv_col = _col(df, tv_cols)
    bv_col = _col(df, bv_cols)
    lv_col = _col(df, lv_cols)

    # 聚合到每个 selection 的最具代表性的一行（成交量最大）
    probs_raw: Dict[str, float] = {}
    total_vol = 0.0

    for _, row in df.iterrows():
        sel = _normalize_selection(row.get(sel_col))
        if not sel:
            continue
        back = _as_float(row.get(back_col), None) if back_col else None
        lay = _as_float(row.get(lay_col), None) if lay_col else None
        # 价格选取：优先买盘；否则卖盘；否则两者均值
        price = back if isinstance(back, (int, float)) and back else (lay if isinstance(lay, (int, float)) and lay else None)
        if price is None and isinstance(back, (int, float)) and isinstance(lay, (int, float)):
            price = (back + lay) / 2.0
        if price is None or price <= 1e-9:
            continue

        # 成交量估计：优先 traded；否则 back+lay；否则 0
        tv = _as_float(row.get(tv_col), 0.0) if tv_col else 0.0
        bv = _as_float(row.get(bv_col), 0.0) if bv_col else 0.0
        lv = _as_float(row.get(lv_col), 0.0) if lv_col else 0.0
        vol = tv if tv and tv > 0 else (bv + lv)
        total_vol += vol if vol and vol > 0 else 0.0

        prob = 1.0 / price
        # 若同一 selection 多行，取成交量最大的一行（或概率较稳的一行）
        if sel not in probs_raw:
            probs_raw[sel] = prob
        else:
            # 简单选择：保留概率更接近中值的，或者使用成交量权重（此处简化为较大成交量优先，但我们未保存 vol/概率对照，保持首行）
            # 这里直接选择：若新概率更接近现有概率，则进行平滑（避免跳跃）
            probs_raw[sel] = (probs_raw[sel] * 0.6) + (prob * 0.4)

    return probs_raw, total_vol


def _extract_wide_format(df: pd.DataFrame) -> Tuple[Dict[str, float], float]:
    """
    宽表结构：一行包含主/平/客的买盘或欧赔列，可能伴随成交量列。
    返回：三路概率（未归一化）与总成交量估计。
    """
    # 买盘价格或欧赔（扩充中文别名）
    h_cols = [
        "home_back_odds", "home_odds", "主胜", "主队", "home", "h",
        "主胜赔率", "欧赔主胜", "主胜(欧指)", "主胜_欧", "初盘主胜", "终盘主胜"
    ]
    d_cols = [
        "draw_back_odds", "draw_odds", "平局", "平", "draw", "d",
        "平局赔率", "欧赔平局", "平局(欧指)", "平局_欧", "初盘平局", "终盘平局"
    ]
    a_cols = [
        "away_back_odds", "away_odds", "客胜", "客队", "away", "a",
        "客胜赔率", "欧赔客胜", "客胜(欧指)", "客胜_欧", "初盘客胜", "终盘客胜"
    ]

    def _first_nonzero_numeric(series) -> Optional[float]:
        for x in series:
            v = _as_float(x, None)
            if isinstance(v, (int, float)) and v and v > 0:
                return float(v)
        return None

    hv = None
    dv = None
    av = None
    for c in h_cols:
        cc = _col(df, [c])
        if cc:
            hv = _first_nonzero_numeric(df[cc])
            break
    for c in d_cols:
        cc = _col(df, [c])
        if cc:
            dv = _first_nonzero_numeric(df[cc])
            break
    for c in a_cols:
        cc = _col(df, [c])
        if cc:
            av = _first_nonzero_numeric(df[cc])
            break

    probs_raw: Dict[str, float] = {}
    if hv and hv > 0:
        probs_raw["home"] = 1.0 / hv
    if dv and dv > 0:
        probs_raw["draw"] = 1.0 / dv
    if av and av > 0:
        probs_raw["away"] = 1.0 / av

    # 成交量（如果存在）
    tv_candidates = [
        "traded_volume", "成交量", "成交", "成交额", "matched",
        "总成交量", "总成交", "总成交额", "总成交金额"
    ]
    tv_col = _col(df, tv_candidates)
    total_vol = 0.0
    if tv_col:
        try:
            # 稳健求和：逐项解析为浮点，过滤非数值
            total_vol = float(pd.to_numeric(df[tv_col].apply(lambda x: _as_float(x, 0.0)), errors="coerce").fillna(0.0).sum())
        except Exception:
            total_vol = 0.0

    return probs_raw, total_vol


def compute_exchange_features_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    """
    从 DataFrame 计算三路概率与流动性权重，并生成融合摘要。
    输出结构：
      {
        "probs": {"home_win": float, "draw": float, "away_win": float},
        "liquidity_weight": float,
        "summary": str
      }
    """
    # 尝试长表；若失败则回退宽表
    probs_raw, total_vol = _extract_long_format(df)
    if not probs_raw or len(probs_raw) < 2:
        probs_raw, total_vol = _extract_wide_format(df)

    # 构建三路概率（若缺失某一项，用最小保护概率），清洗 NaN
    import math
    def _finite_or_zero(v: Any) -> float:
        try:
            x = float(v) if v is not None else 0.0
            return x if math.isfinite(x) and x > 0.0 else 0.0
        except Exception:
            return 0.0
    ph = _finite_or_zero(probs_raw.get("home", 0.0))
    pd_ = _finite_or_zero(probs_raw.get("draw", 0.0))
    pa = _finite_or_zero(probs_raw.get("away", 0.0))

    # 若全部为零，给出均匀分布作为保护
    if ph <= 1e-12 and pd_ <= 1e-12 and pa <= 1e-12:
        ph, pd_, pa = 1.0, 1.0, 1.0

    s = ph + pd_ + pa
    if not math.isfinite(s) or s <= 0.0:
        s = 1.0  # 防御：避免 1e-9 放大，先归一化到均匀或已有项
        # 若至少一项非零，则仅就非零项归一化
        nz = [v for v in [ph, pd_, pa] if v > 0.0]
        if nz:
            s = sum(nz)
    ph /= s
    pd_ /= s
    pa /= s

    # 流动性权重：成交量越大，权重越高。简单的饱和函数。
    # 经验值：以 10,000 为软饱和阈值（可按数据规模调整）
    vol_raw = float(total_vol or 0.0)
    try:
        import math
        vol = max(0.0, vol_raw if math.isfinite(vol_raw) else 0.0)
    except Exception:
        vol = max(0.0, vol_raw)
    w = (vol / (vol + 10000.0)) if vol > 0 else 0.0
    # 清洗非有限权重
    try:
        import math
        w = (w if math.isfinite(w) else 0.0)
    except Exception:
        pass
    w = max(0.0, min(1.0, w))

    # —— 新增：解析指数型辅助特征（必指/赔指/盈亏/凯指/凯差/热指） ——
    def _find_side_metric_column(side_keywords, metric_keywords):
        cols = [str(c).strip().lower() for c in df.columns]
        for c in cols:
            if any(sk in c for sk in side_keywords) and any(mk in c for mk in metric_keywords):
                return c
        return None

    def _triplet(metric_keywords):
        h = _find_side_metric_column(["主", "home", "h"], metric_keywords)
        d = _find_side_metric_column(["平", "draw", "d"], metric_keywords)
        a = _find_side_metric_column(["客", "away", "a"], metric_keywords)
        def get(col):
            try:
                return float(df.iloc[0].get(col)) if col else None
            except Exception:
                return _as_float(df.iloc[0].get(col), None) if col else None
        return {"home": get(h), "draw": get(d), "away": get(a)}

    # 非有限值清洗（NaN/Inf → None）
    import math
    def _finite(v):
        return v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))
    def _sanitize_triplet(t: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        return {k: (float(v) if _finite(v) else None) for k, v in t.items()}

    kelly = _sanitize_triplet(_triplet(["凯指", "凯利", "kelly"]))
    heat = _sanitize_triplet(_triplet(["热指", "热度", "heat"]))
    bf_index = _sanitize_triplet(_triplet(["必指", "必发", "bf"]))
    payout = _sanitize_triplet(_triplet(["赔指", "赔付", "payout", "返还率"]))
    profit_loss = _sanitize_triplet(_triplet(["盈亏", "profit", "loss", "pnl"]))

    def _range(vals):
        vs = [float(v) for v in vals.values() if _finite(v)]
        return (max(vs) - min(vs)) if len(vs) >= 2 else None
    kelly_diff = _range(kelly)
    hvals = [float(v) for v in heat.values() if _finite(v)]
    heat_max = (max(hvals) if hvals else 0.0)

    aux = {
        "kelly_index": kelly,
        "kelly_diff": kelly_diff,
        "heat_index": heat,
        "heat_max": heat_max,
        "bf_index": bf_index,
        "payout_index": payout,
        "profit_loss": profit_loss,
    }

    summary = (
        f"交易所快照倾向：主{ph*100:.1f}%、平{pd_*100:.1f}%、客{pa*100:.1f}%；"
        f"成交量约 {vol:.0f}；融合权重 {w:.2f}"
        + (f"；热指峰值 {heat_max:.1f}" if isinstance(heat_max, (int, float)) and heat_max > 0 else "")
        + (f"；凯差 {kelly_diff:.2f}" if isinstance(kelly_diff, (int, float)) else "")
    )
    return {
        "probs": {"home_win": ph, "draw": pd_, "away_win": pa},
        "liquidity_weight": w,
        "summary": summary,
        "aux_metrics": aux,
        "_debug": {
            "probs_raw": probs_raw,
            "sum": float(ph + pd_ + (pa if math.isfinite(pa) else 0.0)),
            "total_vol": total_vol,
            "norm_check": {
                "ph_before": _finite_or_zero(probs_raw.get("home", 0.0)),
                "pd_before": _finite_or_zero(probs_raw.get("draw", 0.0)),
                "pa_before": _finite_or_zero(probs_raw.get("away", 0.0)),
                "s": s,
                "ph_after": ph,
                "pd_after": pd_,
                "pa_after": pa
            }
        }
    }


def load_exchange_features_from_bytes(buf: bytes, sheet: Optional[Any] = None) -> Dict[str, Any]:
    """从字节读取 Excel 并计算交易所特征（支持 sheet 名或索引）。"""
    df = _read_excel_from_bytes(buf, sheet)
    return compute_exchange_features_from_df(df)


def load_exchange_features_from_path(path: str, sheet: Optional[Any] = None) -> Dict[str, Any]:
    """从文件路径读取 Excel 并计算交易所特征。"""
    if not os.path.isabs(path):
        base_dir = os.path.dirname(__file__)
        path = os.path.abspath(os.path.join(base_dir, "..", "..", "..", path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Exchange Excel not found: {path}")
    sheet_name = 0 if sheet is None else sheet
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    except Exception:
        df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return compute_exchange_features_from_df(df)