#!/bin/bash

# 获取当前脚本所在目录 (HOV-SG 根目录)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# 获取项目根目录 (HOV-SG 的上一级目录, 即 /home/scene)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 设置 PYTHONPATH 确保能找到 hovsg 包
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR

# 定义日志目录
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# 定义需要运行的场景 ID 列表
# SCENES=("4ok" "5cd" "nfv" "tee")
SCENES=("tee")

echo "------------------------------------------------"
echo "脚本位置: $SCRIPT_DIR"
echo "项目根目录: $PROJECT_ROOT"
echo "日志目录: $LOG_DIR"
echo "------------------------------------------------"

# 循环遍历并执行命令
for SCENE in "${SCENES[@]}"
do
    LOG_FILE="$LOG_DIR/create_graph_${SCENE}.log"
    
    echo "------------------------------------------------" | tee -a "$LOG_FILE"
    echo "正在处理场景: $SCENE" | tee -a "$LOG_FILE"
    echo "开始时间: $(date)" | tee -a "$LOG_FILE"
    echo "日志将保存至: $LOG_FILE"
    echo "------------------------------------------------" | tee -a "$LOG_FILE"

    python "$SCRIPT_DIR/application/create_graph.py" \
        main.dataset=goat \
        main.dataset_path="$PROJECT_ROOT/Goat-core/dataset" \
        main.split=. \
        main.scene_id="$SCENE" \
        main.save_path="/root/autodl-tmp/Goat-core/output" \
        pipeline.create_graph=True \
        models.clip.checkpoint="$SCRIPT_DIR/checkpoints/laion2b_s32b_b79k.bin" \
        models.sam.checkpoint="$SCRIPT_DIR/checkpoints/sam_vit_h_4b8939.pth" \
        2>&1 | tee -a "$LOG_FILE"

    echo "------------------------------------------------" | tee -a "$LOG_FILE"
    echo "场景 $SCENE 处理完成！" | tee -a "$LOG_FILE"
    echo "结束时间: $(date)" | tee -a "$LOG_FILE"
    echo "------------------------------------------------" | tee -a "$LOG_FILE"
done

echo "所有任务已执行完毕。请查看 $LOG_DIR 下的日志文件获取详细信息。"
