#!/usr/bin/env bash
# clean.sh — очистка результатов mlf-fuzz
#
# Использование:
#   ./clean.sh            # показывает что будет удалено (dry-run)
#   ./clean.sh --yes      # удаляет без переспрашивания
#   ./clean.sh --yes --keep-model   # удаляет всё, кроме обученной модели

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=1
KEEP_MODEL=0

for arg in "$@"; do
  case "$arg" in
    --yes)        DRY_RUN=0 ;;
    --keep-model) KEEP_MODEL=1 ;;
    --help|-h)
      echo "Использование: $0 [--yes] [--keep-model]"
      echo "  (без флагов)   показать что будет удалено"
      echo "  --yes          удалить без переспрашивания"
      echo "  --keep-model   сохранить /tmp/mlf_model.pkl"
      exit 0
      ;;
    *) echo "Неизвестный флаг: $arg. Используй --help"; exit 1 ;;
  esac
done

# ─── цвета ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; YEL='\033[0;33m'; GRN='\033[0;32m'; RST='\033[0m'

removed=0
skipped=0

remove() {
  local path="$1"
  local label="$2"
  if [[ -e "$path" || -L "$path" ]]; then
    local size
    if [[ -d "$path" ]]; then
      size="$(du -sh "$path" 2>/dev/null | cut -f1)"
    else
      size="$(du -sh "$path" 2>/dev/null | cut -f1)"
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
      echo -e "  ${YEL}[dry-run]${RST}  ${label}  (${size})"
    else
      rm -rf "$path"
      echo -e "  ${RED}удалено ${RST}   ${label}  (${size})"
    fi
    (( removed++ )) || true
  else
    (( skipped++ )) || true
  fi
}

# ─── 1. Выходные директории AFL++ ─────────────────────────────────────────────
echo -e "\n${GRN}── Выходные директории AFL++ ──────────────────────────────${RST}"
shopt -s nullglob

for d in "$SCRIPT_DIR"/out_*/; do
  remove "$d" "$d"
done
for d in "$SCRIPT_DIR"/output_*/; do
  remove "$d" "$d"
done
for d in "$SCRIPT_DIR"/findings/; do
  remove "$d" "$d"
done

# ─── 2. ML тренировочный лог ──────────────────────────────────────────────────
echo -e "\n${GRN}── ML тренировочный лог ───────────────────────────────────${RST}"
MLF_LOG="${MLF_TRAINING_LOG:-/tmp/mlf_training.bin}"
remove "$MLF_LOG" "$MLF_LOG"

# ─── 3. ML модель ─────────────────────────────────────────────────────────────
echo -e "\n${GRN}── ML модель ──────────────────────────────────────────────${RST}"
MLF_MODEL="${MLF_MODEL_PATH:-/tmp/mlf_model.pkl}"
if [[ $KEEP_MODEL -eq 1 ]]; then
  echo -e "  ${GRN}[пропущено]${RST} $MLF_MODEL  (--keep-model)"
else
  remove "$MLF_MODEL" "$MLF_MODEL"
fi

# ─── 4. /tmp мусор ────────────────────────────────────────────────────────────
echo -e "\n${GRN}── /tmp файлы mlf-fuzz ────────────────────────────────────${RST}"
MLF_SOCK="${MLF_SCHEDULER_SOCKET:-/tmp/mlf_scheduler.sock}"
remove "$MLF_SOCK" "$MLF_SOCK"

# Дополнительные /tmp директории используемые в тестах
remove "/tmp/test_in"  "/tmp/test_in"
remove "/tmp/test_out" "/tmp/test_out"

# Временные out_* папки в /tmp
for d in /tmp/out_*/; do
  remove "$d" "$d"
done

# ─── Итог ─────────────────────────────────────────────────────────────────────
echo ""
if [[ $DRY_RUN -eq 1 ]]; then
  echo -e "${YEL}Dry-run: ничего не удалено. Запусти с --yes чтобы удалить.${RST}"
  echo -e "Найдено объектов для удаления: ${removed}"
else
  echo -e "${GRN}Готово. Удалено объектов: ${removed}${RST}"
fi
echo ""
