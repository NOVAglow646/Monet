ckpt_name=9.30_avt_v3_pt_full_wo_maze_latent8_ce4.0_align-wt0.0001_emph-wt1.0
step=200
src=/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v3/${ckpt_name}/checkpoint-${step}
output=/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v3/${ckpt_name}-checkpoint-${step}.zip
include_subdir=False

if [ -z "$src" ] || [ -z "$output" ] || [ -z "$include_subdir" ]; then
    echo "用法: $0 <source_folder> <output_zip> <include_subdir:True|False>"
    exit 1
fi

if [ "$include_subdir" = "True" ]; then
    # 压缩整个目录（包含子目录），只保留最后一级目录名
    (cd "$(dirname "$src")" && zip -r "$output" "$(basename "$src")")
else
    # 只压缩目录下的文件，不包含子目录，保留最后一级目录名
    (cd "$(dirname "$src")" && zip "$output" "$(basename "$src")"/*)
fi

python /mmu_vcg_ssd/shiyang06/Tool/huggingface.py --item /mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v3/${ckpt_name}-checkpoint-${step}.zip

step=400
src=/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v3/${ckpt_name}/checkpoint-${step}
output=/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v3/${ckpt_name}-checkpoint-${step}.zip
if [ -z "$src" ] || [ -z "$output" ] || [ -z "$include_subdir" ]; then
    echo "用法: $0 <source_folder> <output_zip> <include_subdir:True|False>"
    exit 1
fi

if [ "$include_subdir" = "True" ]; then
    # 压缩整个目录（包含子目录），只保留最后一级目录名
    (cd "$(dirname "$src")" && zip -r "$output" "$(basename "$src")")
else
    # 只压缩目录下的文件，不包含子目录，保留最后一级目录名
    (cd "$(dirname "$src")" && zip "$output" "$(basename "$src")"/*)
fi

python /mmu_vcg_ssd/shiyang06/Tool/huggingface.py --item /mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v3/${ckpt_name}-checkpoint-${step}.zip