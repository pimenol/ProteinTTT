# Журнал экспериментов — ProteinTTT_fresh

## 2026-07-29 — Анализ BFVD2: pLDDT ESMFold vs ProteinTTT vs AF (782k белков)

**Гипотеза.**
На BFVD2 (вирусные белки с готовыми MSA, ≤500 aa) test-time training через ProteinTTT
должен поднимать mean pLDDT относительно ESMFold-baseline, особенно там, где baseline
низкий (<45). Ожидаем, что TTT обгонит «лёгкий» AF-pLDDT (`plddt_AF` из summary) на
значимой доле целей — не потому что AF хуже структурно, а потому что ESMFold+TTT
оптимизирует confidence head под конкретную последовательность. Проверяемое следствие:
средний ΔpLDDT > 0, монотонная зависимость «ниже baseline → больше выигрыш», и
кластер «unchanged» (идентичный pLDDT до/после) на высоком baseline — признак
early-stop / confidence-collapse без реального шага TTT.

**Изменения.**
- `scripts/generate_plddt_report.py` — загрузка `bfvd2/plddt_summary.csv`, агрегация
  статистик (means, bins по длине/baseline/pTM, корреляции, top-k), генерация HTML.
- `bfvd2/plddt_analysis.html` — интерактивный отчёт (Chart.js): распределение ΔpLDDT,
  pLDDT по длине, Δ vs baseline bin, scatter ESMFold→TTT, таблицы top improvements
  и случаев где TTT ≫ AF.
- Код пайплайна / конфигов TTT в этом ходе **не менялся** — только артефакты анализа.

**Команда.**
```bash
source .../miniconda3/etc/profile.d/conda.sh && conda activate proteinttt
cd ProteinTTT_fresh
python scripts/generate_plddt_report.py
# вход:  bfvd2/plddt_summary.csv  (782 664 строк, 14 колонок)
# выход: bfvd2/plddt_analysis.html
```

**Job ID.**
Данные собраны array-запусками `scripts/run_dataset.sh` (job-name `bfvdv2`), шарды
**4524081–4578640** (279 задач, `./jobs/df/dataset_<id>.out`). Summary агрегирован
в `bfvd2/plddt_summary.csv` (mtime 2026-07-27). Конфиг: `scripts/config_dataset.yaml`
(lr=0.04, steps=20, ags=64, lora r=64, MSA neighbors, max_len=500).

**Результат vs baseline.**
| Метрика | ESMFold | ProteinTTT | AF | Δ (TTT−ESM) |
|---|---:|---:|---:|---:|
| mean pLDDT | 50.73 | 60.66 | 55.88 | **+9.93** |
| median pLDDT | 48.80 | 60.22 | 58.47 | +7.81 |

- **91.8%** белков улучшены (Δ > 0.01), **8.2%** unchanged (Δ ≈ 0), **0%** регрессий.
- TTT beats AF в **62.7%** случаев; **264 052** белков с TTT−AF > 10 pLDDT.
- ΔpLDDT anti-correlates с baseline: r = **−0.35** (p ≈ 0). По baseline-bin:
  `<35` → +12.9, `35–45` → +13.2, `45–55` → +10.3, `55–65` → +7.6, `>65` → +4.6.
- Unchanged cluster: **63 896** белков, mean baseline pLDDT = **61.2** (early-stop).
- SS shift (ESM→TTT): helix **+1.0%**, sheet **+2.1%**, loop **−3.1%**; r(Δhelix, ΔpLDDT) = 0.33.
- Paired Wilcoxon (ESMFold vs TTT): p ≈ 0 на n = 782 587.
- Top ΔpLDDT: до **+68.4** (напр. W0FZ85: 26.7 → 95.1); TTT−AF до **+67.1**.

**Вывод.**
ProteinTTT систематически поднимает reported pLDDT на BFVD2 (~+10 в среднем), с
наибольшим эффектом на низкоуверенных предсказаниях ESMFold. Это согласуется с
confidence adaptation, а не обязательно с улучшением структуры — **нужна валидация
по TM/lDDT** (в summary есть AF pLDDT, но нет структурных метрик для TTT). Кластер
8% «unchanged» на высоком baseline указывает на срабатывание confidence-collapse /
best-state reset без шага обучения. Следующий шаг: subsample top-Δ vs unchanged для
структурного сравнения (TM между baseline/TTF PDB) и проверка, не «играет» ли TTT
только confidence head.
