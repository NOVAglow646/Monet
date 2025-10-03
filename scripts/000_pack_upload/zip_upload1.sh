src=/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v4/10.2_avt_v4_offline_pt10ep_latent10_ce4.0_align-wt2.0_emph-wt2.0
output=/mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_sft//mmu_vcg_ssd/shiyang06/Project/Latent_Think/checkpoint/avt_v4/10.2_avt_v4_offline_pt10ep_latent10_ce4.0_align-wt2.0_emph-wt2.0.zip
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

