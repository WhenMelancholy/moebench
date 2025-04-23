#!/usr/bin/env bash
#
# fix_broken_so_symlinks.sh
#
# 检测并修复 lib 目录下失效的 .so* 符号链接。
# 用法:
#   ./fix_broken_so_symlinks.sh            # 直接修复
#   ./fix_broken_so_symlinks.sh --dry-run  # 仅打印将要执行的修改
#

set -euo pipefail
DRYRUN=0
[[ "${1:-}" == "--dry-run" ]] && DRYRUN=1

LIBDIR="/n/home08/zkong/.conda/envs/olmo/lib"
cd "$LIBDIR"

echo "📂 扫描目录: $LIBDIR"
echo

# find -type l: 仅列出符号链接；! -e: 目标不存在 (broken)
while IFS= read -r symlink; do
    target=$(readlink "$symlink")               # 原始(相对)目标
    canonical=$(readlink -f "$symlink" || true) # 解析后目标(可能不存在)

    if [[ ! -e "$canonical" ]]; then
        echo "❌ 失效链接: $symlink -> $target"

        # 尝试自动修复共享库
        base=$(basename "$symlink") # e.g. libcurand.so 或 libcurand.so.10
        stem=${base%%.so*}          # e.g. libcurand
        pattern="$stem.so*"         # 匹配所有版本
        # 找到同目录下可用的最高版本文件
        newest=$(ls -1 $pattern 2>/dev/null | sort -V | tail -n1 || true)

        if [[ -n "$newest" && -f "$newest" ]]; then
            echo "   ↪️  修复为: $newest"
            if [[ $DRYRUN -eq 0 ]]; then
                ln -sf "$newest" "$symlink"
            fi
        else
            echo "   ⚠️  未找到可用文件，需手动处理"
        fi
    fi
done < <(find . -maxdepth 1 -type l ! -exec test -e {} \; -print)

[[ $DRYRUN -eq 1 ]] && echo -e "\n(仅 dry‑run，未做任何修改)"
