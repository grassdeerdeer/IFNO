
project_dir="/home3/zyf/yp-nmt"

DATADIR=${project_dir}/data/data_bin/char_total
SAVEDIR=${project_dir}/model/char_total/transformer.yue-zh.search
MODELDIR=${SAVEDIR}/checkpoints
mkdir -p ${SAVEDIR}

SRC=zh_YUE
TGT=zh_CN

export CUDA_VISIBLE_DEVICES=0

for bsz in 1024 512 256;  do 
    for lr in 1e-03;  do 
        for ls in 0.1 0.2 0.3;  do
            for warm in 2500 5000; do
                for dp in 0.2 0.3; do 
                    SAVEDIR_NEW=${SAVEDIR}/bsz${bsz}_lr${lr}_ls${ls}_warm${warm}_dp${dp}
                    mkdir -p ${SAVEDIR_NEW}
                    fairseq-train ${DATADIR} --arch transformer_iwslt_de_en \
	                    --source-lang ${SRC} --target-lang ${TGT}  \
                        --optimizer adam --seed 42 --lr ${lr} --adam-betas '(0.9, 0.998)' \
                        --lr-scheduler inverse_sqrt --batch-size ${bsz}  --dropout ${dp} \
                        --criterion label_smoothed_cross_entropy  --label-smoothing ${ls} \
                        --max-update 40000  --warmup-updates ${warm} --warmup-init-lr '1e-07' \
                        --keep-last-epochs 1 --keep-best-checkpoints 1 --num-workers 6 --patience 10 \
	                    --save-dir ${SAVEDIR_NEW}/checkpoints \
                        --log-format simple --log-interval 1 --log-file ${SAVEDIR_NEW}/train.log
                done
            done
        done
    done
done