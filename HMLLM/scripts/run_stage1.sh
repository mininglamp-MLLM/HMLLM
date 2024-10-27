export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

NNODE=1
NUM_GPUS=8
MASTER_NODE='localhost'

torchrun  --nnodes=${NNODE} --nproc_per_node=${NUM_GPUS} \
    --rdzv_endpoint=${MASTER_NODE}:10068 \
    --rdzv_backend=c10d \
    tasks/train_stage1.py \
    $(dirname $0)/config_stage1.py \
    output_dir ${OUTPUT_DIR}
