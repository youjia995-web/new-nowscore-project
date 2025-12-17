from pydantic_settings import BaseSettings
from pathlib import Path
import os
from dotenv import load_dotenv
from typing import Optional

# 加载 .env（如果存在）
load_dotenv()

class Settings(BaseSettings):
    # DeepSeek API 配置（改为环境变量读取，默认为空以禁用外网调用）
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_api_base: str = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    # 强制要求通过外部 AI API（为 true 时禁用本地回退）
    require_ai_api: bool = os.getenv("REQUIRE_AI_API", "false").lower() == "true"
    
    # 服务器配置
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "3309"))

    # 其他配置
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # 数据库配置：默认为仓库根目录下的 football_analysis.db（绝对路径），可通过环境变量覆盖
    db_path: str = os.getenv(
        "DB_PATH",
        str(Path(__file__).resolve().parents[3] / "football_analysis.db")
    )

    # 阵容数据库：与主库解耦，默认仓库根目录 rosters.db，可通过环境变量覆盖
    roster_db_path: str = os.getenv(
        "ROSTER_DB_PATH",
        str(Path(__file__).resolve().parents[3] / "rosters.db")
    )

    # 阵型数据文件路径（Excel）：默认仓库根目录 formation.xlsx，可通过环境变量覆盖
    formation_xlsx_path: str = os.getenv(
        "FORMATION_XLSX_PATH",
        str(Path(__file__).resolve().parents[3] / "formation.xlsx")
    )

    # 交易所快照文件路径（Excel）：默认仓库根目录 必发交易快照.xlsx，可通过环境变量覆盖
    exchange_xlsx_path: str = os.getenv(
        "EXCHANGE_XLSX_PATH",
        str(Path(__file__).resolve().parents[3] / "必发交易快照.xlsx")
    )

    # 交易所快照回退读取（默认禁用；启用则在未上传时尝试默认 Excel）
    enable_exchange_default_load: bool = os.getenv("ENABLE_EXCHANGE_DEFAULT_LOAD", "false").lower() == "true"

    # 盘口异常判定与评分（可配与标定骨架）
    hcap_threshold_minor: float = float(os.getenv("HCAP_THRESHOLD_MINOR", "0.25"))
    hcap_threshold_major: float = float(os.getenv("HCAP_THRESHOLD_MAJOR", "0.50"))
    hcap_full_score_delta: float = float(os.getenv("HCAP_FULL_SCORE_DELTA", "0.50"))
    hcap_reverse_dir_margin: float = float(os.getenv("HCAP_REVERSE_DIR_MARGIN", "0.10"))
    hcap_water_adj_strength: float = float(os.getenv("HCAP_WATER_ADJ_STRENGTH", "0.20"))
    enable_hcap_quantile_calibration: bool = os.getenv("ENABLE_HCAP_CALIBRATION", "false").lower() == "true"

    # —— 新增：底层概率与融合的可配项 ——
    enable_dc_correlation: bool = os.getenv("ENABLE_DC_CORRELATION", "true").lower() == "true"
    dc_rho: float = float(os.getenv("DC_RHO", "-0.12"))
    enable_draw_boost: bool = os.getenv("ENABLE_DRAW_BOOST", "true").lower() == "true"
    draw_boost_strength: float = float(os.getenv("DRAW_BOOST_STRENGTH", "0.08"))  # 最大增强幅度（约8%）
    fusion_mode: str = os.getenv("FUSION_MODE", "geometric")  # geometric|linear
    # 概率后验校准（等距单调插值），需先运行训练脚本写入参数
    enable_prob_calibration: bool = os.getenv("ENABLE_PROB_CALIBRATION", "true").lower() == "true"

    # —— 新增：贝叶斯引擎与分布扩展 ——
    enable_bayes_engine: bool = os.getenv("ENABLE_BAYES_ENGINE", "false").lower() == "true"
    bayes_params_key: str = os.getenv("BAYES_PARAMS_KEY", "bayes_hier_params_v1")
    bayes_use_nb: bool = os.getenv("BAYES_USE_NB", "false").lower() == "true"
    bayes_use_zip: bool = os.getenv("BAYES_USE_ZIP", "false").lower() == "true"

    # —— 新增：双变量泊松（共享成分） ——
    enable_bivariate_poisson: bool = os.getenv("ENABLE_BIVARIATE_POISSON", "false").lower() == "true"
    # 共享强度（0-1），越大相关性越强，建议 0.10~0.25
    bp_shared_strength: float = float(os.getenv("BP_SHARED_STRENGTH", "0.15"))
    # 动态共享强度：随总预期进球调整相关性（默认启用）
    enable_bp_dynamic: bool = os.getenv("ENABLE_BP_DYNAMIC", "true").lower() == "true"
    bp_total_center: float = float(os.getenv("BP_TOTAL_CENTER", "2.60"))
    bp_total_span: float = float(os.getenv("BP_TOTAL_SPAN", "1.00"))
    bp_total_amp: float = float(os.getenv("BP_TOTAL_AMP", "0.25"))

    # —— 新增：融合后平局对齐校准 ——
    enable_draw_align: bool = os.getenv("ENABLE_DRAW_ALIGN", "true").lower() == "true"
    draw_align_strength: float = float(os.getenv("DRAW_ALIGN_STRENGTH", "0.10"))
    draw_align_mode: str = os.getenv("DRAW_ALIGN_MODE", "market")  # market|league
    league_draw_base: float = float(os.getenv("LEAGUE_DRAW_BASE", "0.27"))

    # —— 新增：联赛节奏与主场优势（影响泊松均值） ——
    league_base_rate: float = float(os.getenv("LEAGUE_BASE_RATE", "1.00"))
    home_advantage: float = float(os.getenv("HOME_ADVANTAGE", "1.05"))
    league_tempo: float = float(os.getenv("LEAGUE_TEMPO", "1.00"))
    # DC 相关性动态化（rho 随预期进球和而衰减）
    enable_dc_rho_dynamic: bool = os.getenv("ENABLE_DC_RHO_DYNAMIC", "true").lower() == "true"
    # —— 历史融合配置（赛季指标） ——
    history_blend_enabled: bool = os.getenv("HISTORY_BLEND_ENABLED", "true").lower() == "true"
    history_blend_max: float = float(os.getenv("HISTORY_BLEND_MAX", "0.50"))
    history_blend_gp_threshold: int = int(os.getenv("HISTORY_BLEND_GP_THRESHOLD", "12"))  # 线性模式阈值（线性降为0所需场次）
    history_default_season: Optional[str] = os.getenv("HISTORY_DEFAULT_SEASON", None)
    # 模式：linear|exp
    history_blend_mode: str = os.getenv("HISTORY_BLEND_MODE", "linear")
    # 指数衰减半衰期（达到权重一半所需场次），例如 6 表示 6 场后权重减半
    history_blend_half_life: int = int(os.getenv("HISTORY_BLEND_HALF_LIFE", "6"))
    # 可选：超过此场次直接截断为 0（0 表示不截断）
    history_blend_cutoff_gp: int = int(os.getenv("HISTORY_BLEND_CUTOFF_GP", "0"))

    gru_enabled_leagues: str = os.getenv("GRU_ENABLED_LEAGUES", "意甲,serie a,serie_ligue")
    gru_model_weight: float = float(os.getenv("GRU_MODEL_WEIGHT", "0.50"))

settings = Settings()