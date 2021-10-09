flow_size = 100
pkt_size = 200
batch_size = 1  # each data will be taken into consideration for updating w
epochs = 1 # modified balanced winnow is considered as a single pass algorithm
t_labels = ['benign', 'attack_bot', 'attack_DDOS',\
            'attack_portscan']
all_labels = ['vectorize_friday/benign', 'attack_bot', 'attack_DDOS',\
            'attack_portscan', 'Benign_Wednesday', 'DOS_SlowHttpTest',\
            'DOS_SlowLoris', 'DOS_Hulk', 'DOS_GoldenEye', 'FTPPatator',\
            'SSHPatator', 'Web_BruteForce', 'Web_XSS']
