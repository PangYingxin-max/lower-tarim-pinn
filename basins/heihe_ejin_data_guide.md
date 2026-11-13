# 第二流域（黑河下游 / 额济纳绿洲）数据获取指南

这份指南只解决一件事：**怎样拿到可以写进论文、经得起审稿人核查的第二流域逐年数据**。
模板见 `second_basin_template.csv`，落位后另存为 `second_basin_filled.csv`。

## 1. 为什么推荐黑河下游 / 额济纳绿洲

它与塔里木河下游的结构最接近：

- 2000 年前后开始统一水量调度，有连续的年度水量调度记录；
- 下游尾闾为额济纳绿洲（东居延海），植被以胡杨、柽柳、沙枣等荒漠河岸林为主；
- 有公开发表的植被恢复、地下水位响应研究可作对照。

已核实（Crossref 可查）的备选引用：

| 文献 | 期刊 / 年份 | DOI |
|---|---|---|
| Zhang S., Ye Z., Chen Y., Xu Y. Vegetation responses to an ecological water conveyance project in the lower reaches of the Heihe River basin | Ecohydrology, 10(6), 2017 | 10.1002/eco.1866 |
| Guo Q., Feng Q., Li J. Environmental changes after ecological water conveyance in the lower reaches of Heihe River | Environmental Geology, 2009 | 10.1007/s00254-008-1641-1 |
| Zhao C., Cheng G., Nan Z. Relationship between vegetation distribution and groundwater level in the lower reaches of Heihe River basin | IGARSS, 2007 | 10.1109/igarss.2007.4423716 |

注意：不要在正文里引用未经核实的文献。上表三条 DOI 于 2026-10-12 逐条对过 Crossref：
10.1002/eco.1866（Ecohydrology 10(6), 2017，作者 Zhang S., Ye Z., Chen Y., Xu Y.）、
10.1007/s00254-008-1641-1（Environmental Geology 58(7): 7605-7616, 2009）、
10.1109/igarss.2007.4423716（IGARSS 2007, 3963-3966）。

## 2. 需要填的六列数据

| 列名 | 含义 | 单位 |
|---|---|---|
| NDVI | 生长季（5-9 月）区域均值 | 无量纲 |
| TVDI | 由 NDVI-LST 特征空间反演 | 0-1 |
| R_eco_water_1e8m3 | 年度生态输水量 | 1e8 m3 |
| Q_total_1e8m3 | 控制断面年径流量 | 1e8 m3 |
| Source_* | 每个数值的出处（作者+年+表号/页码，或机构+报告名+年份） | 文本 |
| Reliability | high / medium / low | 文本 |

**口径必须与塔里木河一致**：同样是生长季平均、同样是河岸带缓冲区、同样用训练期 min-max 归一化。

## 3. 逐年数据从哪里来

### 3.1 NDVI（2000-2023）

推荐 Google Earth Engine，产物可复现：

1. 影像集：`MODIS/061/MOD13Q1`（Terra，16 天，250 m）。
2. 质量过滤：用 `SummaryQA` 或 `pixel_reliability` 只保留好像元。
3. 时间聚合：每年 5-9 月取均值，得到 24 个年度值。
4. 空间单元：额济纳绿洲范围（或黑河下游河道 0-1 km 缓冲带）。

```javascript
var roi = ee.FeatureCollection(...);            // 额济纳绿洲 或 河道缓冲区
var mod = ee.ImageCollection("MODIS/061/MOD13Q1")
            .filterDate("2000-05-01", "2023-09-30")
            .filterBounds(roi);
function yearly(y) {
  var col = mod.filter(ee.Filter.calendarRange(y, y, "year"));
  var ndvi = col.select("NDVI").map(function (img) {
    var qa = img.select("SummaryQA");
    return img.select("NDVI").updateMask(qa.lte(1)).multiply(0.0001);
  });
  return ndvi.mean().reduceRegion({
    reducer: ee.Reducer.mean(), geometry: roi, scale: 250, maxPixels: 1e9
  }).get("NDVI");
}
var years = ee.List.sequence(2000, 2023);
print(years.map(yearly));
```

备选：NASA LP DAAC 直接下载 MOD13Q1/MOD13A3 后本地处理（可追溯性更强，适合写进方法）。

### 3.2 TVDI

TVDI 不是现成产品，必须自己算：

1. 地表温度用 `MODIS/061/MOD11A2`（8 天，1 km），NDVI 用 MOD13Q1。
2. 按 Sandholt et al. (2002) 在 NDVI-LST 散点图上拟合干边与湿边，**逐年或至少分季节拟合**。
3. `TVDI = (Ts - Ts_min) / (Ts_max - Ts_min)`，其中 Ts_max 为干边、Ts_min 为湿边。
4. 对每个像元算完再取区域均值，不要先取均值再算 TVDI。

湿边样本要包含水体/灌溉区像元，干边样本要落在裸地上；这一步的口径要在方法和补充材料里写清楚。

### 3.2.1 已核实可访问的数据门户（2026-10-12 实测）

| 门户 | 主要用途 | 访问状态 |
|---|---|---|
| 国家青藏高原科学数据中心 data.tpdc.ac.cn | 黑河流域水文、气象、地下水、遥感数据集（含黑河计划系列数据） | 可访问（HTTP 200） |
| 中国气象数据网 data.cma.cn | 额济纳旗、鼎新等站点的气温、降水、蒸发，用于解释 TVDI 年际变化 | 可访问（HTTP 200） |
| 内蒙古自治区水利厅 slt.nmg.gov.cn | 额济纳绿洲生态输水、东居延海水面与水量通报 | 可访问（HTTP 200） |
| 甘肃省水利厅 slt.gansu.gov.cn | 莺落峡、正义峡下泄水量与年度水资源公报 | 服务在线（返回 412，需用浏览器访问） |
| 黑河计划数据管理中心、寒区旱区科学数据中心 | 黑河历史观测数据集 | 本机 DNS 未能解析，未核实，不作为正式引用来源 |

建议的取数顺序：先在 data.tpdc.ac.cn 检索“黑河”数据集，优先下载带 DOI 的数据集（数据集本身可引用，
比引用网页更稳）；再用省级水利厅公报补齐逐年输水量；最后用已发表论文的数据表交叉核对。
每一步都记录数据集名称、版本、DOI，或公报的年份与表号。

找不到的年份就留空，并在 limitations 里写明缺哪几年；不要用相邻年份插值后当成观测值。

### 3.3 生态输水量（R）与总径流量（Q）

黑河下游的水量调度链条是：莺落峡 -> 正义峡 -> 哨马营 -> 狼心山 -> 额济纳绿洲（东居延海）。

优先来源（按可信度排序）：

1. 黑河流域管理局（水利部黄河水利委员会下属）年度水量调度公报 / 年度报告；
2. 《黑河流域水资源公报》与地方水利部门年度公报；
3. 水文年鉴中正义峡、哨马营、狼心山断面的年径流量；
4. 已发表论文里的数据表（引用时写明表号，可靠性标为 medium）。

检索关键词：黑河水量调度、正义峡下泄水量、狼心山、额济纳绿洲、东居延海、额济纳旗生态监测、
黑河流域管理局年度报告；英文可用 Heihe River, Ejina Oasis, Zhengyixia, Langxinshan, East Juyan Lake。

机构主页：黄河水利委员会、甘肃省水利厅、内蒙古自治区水利厅、额济纳旗水务局。

## 4. 落位与运行

```powershell
# 1) 填好 CSV 后先做质量审计
python data_quality_report.py --csv basins/second_basin_filled.csv --outdir reports_quality_second_basin --label "Heihe lower reaches"

# 2) 转成 某个版本.py 需要的三个 Excel
python basins/csv_to_excel.py --csv basins/second_basin_filled.csv --outdir basins/heihe_excel --basin "Heihe lower reaches"

# 3) 用与塔里木河完全相同的流程训练与检验
python v1.py --ndvi basins/heihe_excel/NDVI.xlsx --water basins/heihe_excel/R.xlsx --tvdi basins/heihe_excel/TVDI.xlsx --outdir results_heihe
```

第二流域跑通后，才能把正文里的 single basin 升级为 external validation，并把结果写进补充材料。

## 5. 硬性要求

- **不要用语言模型生成任何数值。** 缺失年份就留空，并在 limitations 里写明。
- 每个数值都要能追到具体表号/页码；做不到就标 low。
- 如果某年只有趋势预测值，标 trend projection 并给 low，不要混进观测序列。
- 第二流域的 NDVI/TVDI 口径必须与塔里木河一致，否则不能叫 external validation。
