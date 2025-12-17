# 交易所快照上传使用说明与常见报错排查

## 使用前置
- 后端服务运行在 `http://127.0.0.1:5518`（例如：`arch -x86_64 ./venv_x86/bin/python -m uvicorn main:app --host 127.0.0.1 --port 5518 --reload`）。
- 前端静态页面通过本地服务器访问，例如：`http://localhost:5531/advanced.html?api=http://127.0.0.1:5518`。
  - 关键点：URL 中必须带 `api` 参数指向后端，否则页面会把请求发送到静态服务器导致 405 错误。

## 操作步骤
1. 打开高级页：`http://localhost:5531/advanced.html?api=http://127.0.0.1:5518`。
2. 在“球队数据”中输入主队与客队名称（用于建立缓存键）。
3. 在“交易所快照”选择框中选取 Excel 文件（例如根目录下的 `必发交易快照.xlsx`）。
4. 点击“上传”，等待提示。
5. 成功后将显示“上传成功”，并在下方弹出解析摘要。

## 缓存与预测联动
- 上传接口会解析 Excel 并生成交易所特征，按事件键缓存到后端内存：`EXCHANGE_CACHE[event_key]`。
- 事件键默认为 `home|away`，其中 `home/away` 为经过球队名解析后的规范名（若解析失败则使用传入原名）。
- 在预测接口中，会尝试读取相同 `event_key` 的交易所特征进行融合；默认情况下，若缓存缺失不会回退加载。只有当启用环境变量 `ENABLE_EXCHANGE_DEFAULT_LOAD=true` 时，后端才会尝试从 `settings.exchange_xlsx_path`（默认 `必发交易快照.xlsx`）加载。

## Sheet 选择
- 可选查询参数 `sheet` 支持工作表名或索引（例如 `sheet=0`）。
- 未指定 `sheet` 时，系统默认读取首个工作表（索引 0），避免返回 `dict` 结构导致解析失败。

## 接口规范
- 路径：`POST /api/admin/exchange-snapshot/upload`
- 表单体：`multipart/form-data`，字段 `file` 为 Excel 文件。
- 查询参数：`home`（主队名）、`away`（客队名）、`event_key`（可选自定义）、`sheet`（可选工作表）。
- 返回：`{ ok: true, event_key, features }`，其中 `features.summary` 为人类可读摘要。

## Curl 示例

上传并使用英文队名：
```bash
curl -i -X POST -H "Expect:" \
  -F "file=@/Users/huojin/Desktop/通用型足彩分析/必发交易快照.xlsx" \
  "http://127.0.0.1:5518/api/admin/exchange-snapshot/upload?home=Liverpool&away=Man%20City"
```

上传并使用中文队名：
```bash
curl -i -X POST -H "Expect:" \
  -F "file=@/Users/huojin/Desktop/通用型足彩分析/必发交易快照.xlsx" \
  "http://127.0.0.1:5518/api/admin/exchange-snapshot/upload?home=利物浦&away=曼城"
```

指定工作表名或索引：
```bash
curl -i -X POST -H "Expect:" \
  -F "file=@/path/to/excel.xlsx" \
  "http://127.0.0.1:5518/api/admin/exchange-snapshot/upload?home=主队&away=客队&sheet=0"
```

## 常见报错与排查

- 405 Method Not Allowed：
  - 症状：页面弹窗显示上传失败 405。
  - 原因：请求发到了静态服务器（如 `http.server`），而非 FastAPI 后端。
  - 解决：确保访问页面时带 `api` 参数指向后端，例如 `?api=http://127.0.0.1:5518`。

- 400 `'dict' object has no attribute 'columns'`：
  - 症状：后端解析 Excel 报上述错误。
  - 原因：当未指定 `sheet` 且 Excel 有多个工作表时，`pandas.read_excel` 可能返回 `dict`。
  - 解决：当前版本已默认读取首个工作表；如仍遇到问题，明确传入 `sheet=0` 或指定具体工作表名。

- 400 缺少 `openpyxl`：
  - 症状：提示需要 Excel 引擎。
  - 解决：安装依赖 `pip install openpyxl`。若需持久化，可将其加入 `backend/requirements.txt` 并在虚拟环境中安装。

- `Invalid HTTP request received`：
  - 症状：curl 返回该错误。
  - 原因：目标端口不是后端 HTTP 服务，或 URL 拼写错误。
  - 解决：确认后端运行在 `127.0.0.1:5518`，检查 URL 与端口。

- 415 Unsupported Media Type / 413 Request Entity Too Large：
  - 症状：上传失败并返回上述状态码。
  - 原因：请求头或体不符合要求、文件过大。
  - 解决：确保使用 `multipart/form-data` 且字段名为 `file`；选择合理大小的 Excel 文件。

## 备注
- 若希望在预测前无需手动上传，可将默认 Excel 放在仓库根目录并命名为 `必发交易快照.xlsx`。注意：该“缓存缺失回退自动加载”默认关闭，如需启用请设置 `ENABLE_EXCHANGE_DEFAULT_LOAD=true`。