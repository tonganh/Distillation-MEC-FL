# python main_mobile.py --edge_update_frequency 3 --task mnist_cnum300_dist8_skew0.8_seed0  --sample uniform --model mlp --algorithm h_fedprox --num_edges 10 --num_clients 300 --num_rounds 300 --num_epochs 5 --learning_rate 0.001 --momentum 0.99  --weight_decay 1e-6 --proportion 0.5 --batch_size 64 --eval_interval 1 --server_gpu_id 0 --num_threads 1 --remove_client 1  --p_move 0.5
python main_mobile.py --edge_update_frequency 6 --task mnist_cnum300_dist8_skew0.8_seed0  --sample uniform --model mlp --algorithm h_fedprox --num_edges 10 --num_clients 300 --num_rounds 300 --num_epochs 5 --learning_rate 0.001 --momentum 0.99  --weight_decay 1e-6 --proportion 0.5 --batch_size 64 --eval_interval 1 --server_gpu_id 0 --num_threads 1 --remove_client 1  --p_move 0.5
python main_mobile.py --edge_update_frequency 9 --task mnist_cnum300_dist8_skew0.8_seed0  --sample uniform --model mlp --algorithm h_fedprox --num_edges 10 --num_clients 300 --num_rounds 300 --num_epochs 5 --learning_rate 0.001 --momentum 0.99  --weight_decay 1e-6 --proportion 0.5 --batch_size 64 --eval_interval 1 --server_gpu_id 0 --num_threads 1 --remove_client 1  --p_move 0.5
