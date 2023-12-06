python main_distill.py --edge_update_frequency 3 --algorithm fed_distill_kip --kip_support_size 10 \
 --distill_iters 3000 --distill_data_path "distill_data_kip/"  --model resnet9_custom --weight_decay 1e-3 \
 --task mnist_cnum100_dist8_skew0.8_seed0 --mean_velocity 50 --std_velocity 10  --sample uniform --num_edges 5 \
 --num_clients 100 --std_num_clients 10 --num_rounds 200 --num_epochs 5 --learning_rate 0.01 \
 --momentum 0.9 --proportion 0.3 \
 --batch_size 64 --eval_interval 1 --gpu 0 --server_gpu_id 0 --num_threads 1 --learning_rate_decay 0.9 --distill_ipc 20 --dropout_value=0.3 --remove_client 1
