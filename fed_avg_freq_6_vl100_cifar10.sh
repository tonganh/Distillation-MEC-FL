python main_mobile.py --edge_update_frequency 6 --task cifar10_cnum100_dist8_skew0.8_seed0 --mean_velocity 100 --std_velocity 2  --sample uniform --model resnet9 --algorithm fed_edgeavg --num_edges 5 --num_clients 100 --std_num_clients 10 --num_rounds 200 --num_epochs 5 --learning_rate 0.0005 --momentum 0.9  --weight_decay 1e-4 --proportion 0.2 --batch_size 64 --eval_interval 1 --server_gpu_id 1 --num_threads 1 --remove_client 1