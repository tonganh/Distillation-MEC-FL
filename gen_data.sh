# python generate_fedtask.py --dataset mnist --dist 0 --skew 0 --num_clients 100
# python generate_fedtask.py --dataset mnist --dist 1 --skew 0.2 --num_clients 60
# python generate_fedtask.py --dataset cifar10 --dist 8 --skew 0.8 --num_clients 200
# rm -rf /home/cong/anhtn/data-distill-fl/fedtask/cifar100_cnum100_dist8_skew0.8_seed0 && \
#     python generate_fedtask.py --dataset cifar100 --dist 8 --skew 0.8 --num_clients 100
# python generate_fedtask.py --dataset mnist --dist 7 --skew 0.8 --num_clients 100
# python generate_fedtask.py --dataset mnist --dist 2 --skew 0.1 --num_clients 100
# python generate_fedtask.py --dataset cifar100 --dist 2 --skew 0.1 --num_clients 100
# python generate_fedtask.py --dataset mnist --dist 8 --skew 0.8 --num_clients 300

# python generate_fedtask.py --dataset cifar10 --dist 8 --skew 0.8 --num_clients 300
# python generate_fedtask.py --dataset cifar100 --dist 8 --skew 0.8 --num_clients 300

# python generate_fedtask.py --dataset mnist --dist 2 --skew 0.1 --num_clients 300


python generate_fedtask.py --dataset fashion_mnist --dist 2 --skew 0.1 --num_clients 300
# python generate_fedtask.py --dataset cifar100 --dist 8 --skew 0.8 --num_clients 300 --number_class_per_client 5
python generate_fedtask.py --dataset octmnist --dist 9 --skew 0.8 --num_clients 300 --number_class_per_client 3

python generate_fedtask.py --dataset cifar100 --dist 2 --skew 0.1 --num_clients 100
