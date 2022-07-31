# 安裝步驟

1. create .env檔並輸入
    ```
    MODEL_SAVE_PATH = 'models/save/'
    PRETRAIN_DIR_PATH = 'data/bert.small.torch/'
    PARAMETER_PATH = 'data/parameter/small/parameters.txt'
    DATASET_PATH = 'dataset/reviews_small.csv'

    MAX_LEN = 512
    SPLIT_RATE = 0.9
    BATCH_SIZE = 1
    LR = 1e-4
    NUM_EPOCHS = 1
    DROPOUT = 0.1
    VOCAB_SIZE = 60005
    GPU_NUMS = 1

    NUM_HIDDENS = 256  #small:256 base:768
    FFN_NUM_INPUT = 256 #small:[256] base:[768]
    FFN_NUM_HIDDENS = 512 #small:512 base:3072
    NORM_SHAPE = 256 #small:[256] base:[768]
    NUM_HEADS = 4 #small:4 base:12
    NUM_LAYERS = 2 #small:2 base:12

    ERROR_RATE = 0.25
    FLIP_START = 0
    FLIP_END = 32

    HISTORY_PATH = history.txt
    ```

2. 下載以下函示庫
    ```bash
    pip3 install torch -i  https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --trusted-host pypi.org
    pip3 install python-dotenv -i  https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --trusted-host pypi.org
    pip3 install pandas -i  https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --trusted-host pypi.org
    pip3 install progressbar -i  https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --trusted-host pypi.org
    ```

3. python download.py

4. python models/preprocess/parameter.py

5. 到dataset目錄裡執行
    - 1000000 
        ```bash
        wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vyiIumFcIds3a2jgvu9Ehoe6hYa3Xnbj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vyiIumFcIds3a2jgvu9Ehoe6hYa3Xnbj" -O 'reviews_medium.zip' && rm -rf /tmp/cookies.txt
        ```
        ```bash
        unzip reviews_medium.zip
        ```
    - 2500000
        ```bash
        wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1j0etchuli5Yh3sw3xHZiLTjJCSX1Xo2m' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1j0etchuli5Yh3sw3xHZiLTjJCSX1Xo2m" -O 'reviews_2500000.zip' && rm -rf /tmp/cookies.txt
        ```
        ```bash
        unzip reviews_medium.zip
        ```