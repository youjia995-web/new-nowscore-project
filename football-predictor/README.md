# 足球比赛预测系统（football-predictor）

面向职业足球的比赛分析与预测系统。整合：
- 定量模型（泊松分布、比分概率矩阵）
- 市场共识（多博彩公司赔率、隐含概率去水与聚合）
- 历史信号与校准（本地 SQLite，支持赛季融合曲线）
- 可选 AI 分析（DeepSeek，可关闭或强制）

后端提供 API（FastAPI），同时挂载静态前端页面（Bootstrap/Chart.js），支持同源访问与一键预览。

---

## 功能特性
- 比赛预测：输入主客队与公司赔率，生成胜平负概率、热门比分、盘口合理性与异象评分。
- 市场融合：按稳定性与公司数量动态融合模型与市场概率，保证概率非负且归一。
- 历史融合：按比赛场次对上季 xG/xGA/xPTS 进行线性/指数权重融合，支持阈值与半衰期配置。
- 校准与回测：提供摘要指标（准确率、Brier、LogLoss、ECE）与分段校准细节。
- 数据录入：
  - 球队赛季统计（xG/xGA/xPTS/PPDA/ODC…）增改查。
  - 比赛记录与公司赔率录入：优先写入原库 `matches_data`，缺失时回退 `matches_manual`。
- 前端：`index.html`（预测页）与 `advanced.html`（扩展视图）。
- 文档：FastAPI 交互式文档（`/docs`）。

---

## 架构说明
- 后端：FastAPI（`backend/main.py`）
  - 路由：`backend/src/api/routes.py`
  - 配置：`backend/src/config.py`（环境变量）
  - 引擎：`backend/src/engines/*`（泊松/市场/历史）
  - 模型：`backend/src/models/match.py`（Pydantic 数据结构）
  - 服务：`backend/src/services/prediction_store.py`（SQLite 读写与表管理）
- 前端：`frontend/index.html`、`frontend/advanced.html`（由后端根路径挂载）
- 数据库：SQLite（默认路径 `football_analysis.db`，可用 `DB_PATH` 覆盖）

后端启动时：
- 开启 CORS（默认允许所有域；生产建议限定）
- 挂载 `frontend/` 到 `/`，前端可同源访问 `/api/*`

---

## 目录结构
- `backend/`
  - `main.py`：后端入口（创建应用、挂载前端、运行 Uvicorn）
  - `requirements.txt`：Python 依赖
  - `src/`
    - `api/routes.py`：接口路由
    - `config.py`：环境变量与默认配置
    - `engines/`：`poisson_engine.py`、`market_engine.py`、`historical_market_engine.py` 等
    - `models/match.py`：请求/响应模型
    - `services/prediction_store.py`：数据库服务
- `frontend/`
  - `index.html`、`advanced.html`、`styles.css`
- `football_analysis.db`：默认 SQLite 数据库
- `.env`：本地环境变量（可选）

---

## 环境要求
- Python 3.10–3.13（推荐 3.11/3.12）
- macOS（命令以 macOS 为例）
- 可选：`sqlite3` CLI（查看与维护数据库）
- 可选：DeepSeek API Key（启用外部 AI 时需要）

注意：在 Apple Silicon（ARM）上，若出现 `pydantic_core` 架构不匹配（x86_64 vs arm64），见文末“故障排除”。

---

## 快速开始

### 1. 创建与激活虚拟环境（可选）
```bash
python3 -m venv backend/venv
source backend/venv/bin/activate
```

### 2. 安装依赖
```bash
pip install -r backend/requirements.txt
```

### 3. 启动后端（同时挂载前端）
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 3309 --reload
```

启动后：
- 前端页面：`http://localhost:3309/index.html`
- 接口文档：`http://localhost:3309/docs`

示例预测调用：
```bash
curl -s http://localhost:3309/api/predict -X POST \
  -H "Content-Type: application/json" \
  -d @tmp_predict.json | jq .
```

---

## 配置说明（环境变量）
- 核心服务：
  - `HOST`（默认 `0.0.0.0`）
  - `PORT`（默认 `3309`）
  - `DEBUG`（默认 `true`）
  - `DB_PATH`（默认仓库根目录 `football_analysis.db`）
- AI 配置：
  - `DEEPSEEK_API_KEY`（留空则禁用外部 AI）
  - `DEEPSEEK_API_BASE`（默认 `https://api.deepseek.com/v1`）
  - `REQUIRE_AI_API`（`true` 时必须配置 Key，否则 503）
- 模型/融合与历史：
  - `FUSION_MODE`：`geometric|linear`（默认 `geometric`）
  - `ENABLE_DC_CORRELATION`、`DC_RHO`、`ENABLE_DC_RHO_DYNAMIC`
  - `ENABLE_DRAW_BOOST`、`DRAW_BOOST_STRENGTH`
  - `HISTORY_BLEND_ENABLED`、`HISTORY_BLEND_MODE`（`linear|exp`）
  - `HISTORY_BLEND_MAX`、`HISTORY_BLEND_GP_THRESHOLD`、`HISTORY_BLEND_HALF_LIFE`、`HISTORY_BLEND_CUTOFF_GP`
- 盘口异常与评分：
  - `HCAP_THRESHOLD_MINOR|MAJOR|FULL_SCORE_DELTA|REVERSE_DIR_MARGIN|WATER_ADJ_STRENGTH`
  - `ENABLE_HCAP_CALIBRATION`（分位标定开关，后续可接历史库）

示例（临时覆盖端口与数据库）：
```bash
export PORT=5517
export DB_PATH="$(pwd)/../football_analysis.db"
```

---

## 前端使用
- 预测页（`index.html`）：填写主客队与赔率，提交后展示模型/市场概率、比分热力图、盘口判定、AI 分析与历史摘要。
- 扩展页（`advanced.html`）：用于查看更详细的历史统计或调试视图（可扩展）。

---

## 接口文档（后端 API）

前缀：`/api`

- 快速探针：
  - `GET /mock-data`：返回三组示例输入（用于前端快速演示）
- 比赛预测：
  - `POST /predict`：请求体为 `MatchInput`，响应为 `PredictionResult`（见 `models/match.py`）
- 市场与回测：
  - `GET /companies`：常见公司列表（静态兜底）
  - `GET /backtest/summary`：回测摘要指标
  - `GET /backtest/calibration`：分段校准明细
- 历史与赛季统计：
  - `GET /admin/teams`：赛季统计查询（支持 team/season/league）
  - `POST /admin/teams`：赛季统计 UPSERT（`UNIQUE(team,season)`）
  - `POST /admin/matches`：比赛记录写入（原库优先，否则回退）
- 指标数据导入/查询：
  - `GET /admin/team-metrics/template?format=xlsx|csv&style=compact|standard`：下载模板
  - `POST /admin/team-metrics/import`：批量导入每日指标（覆盖文件中赛季后再导入）
- 别名映射：
  - `POST /admin/team-aliases/import`：导入队名别名（中文/英文/简称，含可选联赛/语言）
- 预测日志：
  - `GET /predictions`、`GET /predictions/{pid}`、`PATCH /predictions/{pid}/result`

示例：查询赛季统计
```bash
curl "http://localhost:3309/api/admin/teams?team=利物浦&season=2024-2025&limit=20&offset=0"
```

示例：赛季统计 UPSERT
```bash
curl -X POST "http://localhost:3309/api/admin/teams" \
  -H "Content-Type: application/json" \
  -d '{"team":"曼城","season":"2024-2025","xg":85.2,"xga":28.9,"xpts":88.5,"notes":"模型基线"}'
```

---

## 数据库说明

默认路径：仓库根目录 `football_analysis.db`（可通过 `DB_PATH` 覆盖）。

主要表（观测）：
- `matches_data`：原始比赛与赔率观测
- `team_season_stats`：球队赛季统计
- `company_weights`：公司权重（自学习）
- `prediction_logs`：预测日志
- `matches_manual`：手动录入比赛

常用命令：
```bash
sqlite3 "$DB_PATH" ".tables"
sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM matches_data;"
```

说明：
- `POST /api/admin/matches` 会在检测到 `matches_data` 表存在时自动映射核心列（赛季/联赛/轮次/队名/排名/半全场/公司赔率），否则写入 `matches_manual`。
- 历史引擎自动识别中英文列名与别名，生成用于模型校正的历史信号与摘要。

---

## 脚本与工具
- `backend/src/scripts/import_excel_matches.py`：从 Excel 导入比赛到 SQLite（支持常见中文/英文列名，自动识别与填充）。
  - 示例：
    ```bash
    python3 backend/src/scripts/import_excel_matches.py /path/英超2024-2025.xlsx --league 英超 --season 2024-2025
    ```
- `backend/src/scripts/calibrate_league_params.py`：联赛参数最小标定（不依赖市场赔率，基于泊松引擎网格搜索）。
  - 示例：
    ```bash
    python3 backend/src/scripts/calibrate_league_params.py --season 2024-2025 --league 英超
    ```

---

## 部署与运维

开发运行：
```bash
cd backend && uvicorn main:app --host 0.0.0.0 --port 3309 --reload
```

生产建议：
- 将 CORS 限定到可信域名
- 通过 `uvicorn`/`gunicorn` + 守护进程 + 反向代理（Nginx）
- SQLite 单实例或迁移外部数据库（接口已抽象）
- 保护 `.env` 与数据库文件权限

---

## 故障排除（macOS / Apple Silicon）

### pydantic_core 架构不匹配（x86_64 vs arm64）
- 现象：启动时报错 `mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')`。
- 解决方案A（推荐）：使用 Python 3.11/3.12 重新创建 venv 并安装依赖（优先拉取兼容 wheel）。
- 解决方案B（临时兼容）：在 Apple Silicon 上使用 Rosetta 运行 x86_64 解释器：
  ```bash
  cd backend
  arch -x86_64 ./venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 5518 --reload
  ```
- 解决方案C（需要编译器）：强制从源码编译 `pydantic-core` 的 `arm64` 版本（需安装 Rust 等工具链）：
  ```bash
  pip install --force-reinstall --no-binary pydantic-core pydantic-core
  ```

### 数据库路径与多文件问题
- 症状：仓库根与 `backend/` 下出现多个 `football_analysis.db`。
- 建议：在 `backend/.env` 设置绝对路径 `DB_PATH` 指向统一数据库文件。

### 端口占用或无法访问 `/docs`
- 检查端口是否被占用；确认服务已在期望端口启动。
- 访问 `http://localhost:3309/docs`（或你自定义的端口）。

### AI 分析 503
- 若 `REQUIRE_AI_API=true` 且未配置 `DEEPSEEK_API_KEY`，后端会返回 503。
- 清空或关闭该开关以使用本地 fallback 文本分析。

---

## 版本与后续规划（简要）
- 增强公司权重与市场聚合的一致性校验（非负剪裁与归一），减少异常概率分布。
- 支持联赛级历史融合与市场融合曲线（可通过配置或后续标定服务下发）。
- 管理端 API（/admin）持续完善，便于前端录入与参数调整。


---

## 变更记录
- 2025-10-23
  - 新增管理员录入页面：球队赛季统计（增改查）与比赛记录（含公司赔率）
  - 新增接口：`GET/POST /api/admin/teams`、`POST /api/admin/matches`、`GET /api/companies`、`GET /api/backtest/*`
  - 前端接入后端查询并完成表格渲染；挂载前端到根路径
  - 使用本地 SQLite 历史库；完成 UPSERT 行为与历史引擎列映射

---

## 许可证
暂未设置开源许可证（默认保留所有权利）。如需开放，请补充许可证文件（如 MIT/Apache-2.0）。