#!/bin/bash

# 定义需要运行的场景 ID 列表
SCENES=("4ok" "5cd" "nfv" "tee")

# 获取当前工作目录
CUR_DIR=$(pwd)

# 循环遍历并执行命令
for SCENE in "${SCENES[@]}"
do
    echo "------------------------------------------------"
    echo "正在处理场景: $SCENE"
    echo "------------------------------------------------"

    python HOV-SG/application/create_graph.py \
        main.dataset=goat \
        main.dataset_path="$CUR_DIR/Goat-core/dataset" \
        main.split=. \
        main.scene_id="$SCENE" \
        main.save_path="$CUR_DIR/Goat-core/output" \
        pipeline.create_graph=True \
        models.clip.checkpoint=/home/scene/HOV-SG/checkpoints/laion2b_s32b_b79k.bin \
        models.sam.checkpoint=/home/scene/HOV-SG/checkpoints/sam_vit_h_4b8939.pth

    echo "场景 $SCENE 处理完成！"
done

echo "所有任务已执行完毕。"