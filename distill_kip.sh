python main_distill.py --edge_update_frequency 3 --algorithm fed_distill_kip --kip_support_size 10  --distill_iters 3000 \
    --distill_data_path "distill_data_kip/"  --model resnet18 --weight_decay 1e-6  --task fashion_mnist_cnum300_dist8_skew0.8_seed0 \
    --sample uniform --num_edges 10  --num_clients 100  --num_rounds 300 --num_epochs 5 --learning_rate 0.01  --momentum 0.9 --proportion 0.5  \
    --batch_size 64 --eval_interval 1 --gpu 0 --server_gpu_id 0 --num_threads 1  --learning_rate_decay 0.9 --distill_ipc 10 --dropout_value=0.3 \
     --architec_KIP "Conv" --depth_KIP 1