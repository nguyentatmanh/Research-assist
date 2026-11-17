#!/usr/bin/env bash
set -e

# ==============================================================================
# Research Assist - Kaggle Bootstrap Launcher
# ==============================================================================

PROJECT_DIR="${PROJECT_DIR:-/kaggle/working/research-assist}"
REPO_URL="${REPO_URL:-https://github.com/nguyentatmanh/Research-assist.git}"

if [ -f "kaggle/run_kaggle.py" ]; then
    PROJECT_DIR="$(pwd)"
    echo "✔ Sử dụng mã nguồn Research Assist hiện tại tại $PROJECT_DIR"
elif [ -d "$PROJECT_DIR" ] && [ -f "$PROJECT_DIR/kaggle/run_kaggle.py" ]; then
    echo "✔ Sử dụng mã nguồn Research Assist tại $PROJECT_DIR"
else
    echo "📥 Đang tải mã nguồn Research Assist từ $REPO_URL..."
    git clone -q "$REPO_URL" "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH:-}"
exec python3 -u -m kaggle.run_kaggle "$@"
