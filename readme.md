### Usage

-------
建立docker image
```cmd
$ git clone https://github.com/xxxxsars/trt_benchmark
$ cd trt_benchmark
$ docker buid -t trt_benchmart .
```

執行docker container,並export ssh port
```cmd
$ nvidia-docker run -d -p 8022:22 trt_benchmark
```

利用ssh連線並進行測試
```cd
$ ssh root@localhost -p 8022
$ cd /tmp/trt_benchmark
#產生onnx
$ python gen_onnx.py
#測試tensorRT benchmark
$ python benchmark_trt.py
#一般model benchmark
$ python benchmark.py
```

### Benchmark Image Classification

-------


**PC配置**
* CPU: Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz


* GPU: GEFORCE® GTX 1080 Ti 11 GB GDDR5X (3584 CUDA core) 


* RAM: 4G DDR4 @ 2133MHz x 4


* Hard Disk: WD SSD 1TB 

**JetSon NX配置**
* CPU: 6-core NVIDIA Carmel 64-bit ARMv8.2 @ 1.40GHz


* GPU: 384 CUDA core NVIDIA Volta @ 1100MHz with 48 Tensor Cores


* RAM: 8GB LPDDR4x @ 1600MHz x 1


* Hard Disk: SanDisk ExtremePRO microSDXC 64GB

### 演算法選擇

-------

選用kears演算法中，訓練樣本為224x224圖片的演算法來進行比較。

演算法的準確率與訓練層數如下表：
 
| Ｍodel       | Accuracy | Depth |
|-------------|----------|-------|
| VGG16       | 0.901    | 23    |
| VGG19       | 0.900    | 26    |
| ResNet50    | 0.921    | 50    |
| ResNet101   | 0.928    | 101   |
| DenseNet121 | 0.923    | 121   |
| MobileNet   | 0.895    | 88    |


### Testing Benchmark

-------

測試分為三組
1. PC平台 (GTX 1080 Ti)
2. Jetson NX **[不使用]** TensorRT
3. Jetson NX  **[使用]** TensorRT

測試方法:


以上都使用kears預先訓練好的模型，並使用不同演算法對一張大象照片進行影像分類，以下為各平台的預測時間(單位為**[秒]**)。

| Model     | GTX 1080 Ti | Jetson NX (with TensorRT) | Jetson NX (without TensorRT) |
|-----------|-------------|-------------|---------------------------|
| VGG16     | 5.27        | 0.00342     | 11.22                     |
| VGG19     | 3.03        | 0.00341     | 11.71                     |
| ResNet50  | 3.48        | 0.00581     | 9.85                      |
| ResNet101 | 4.14        | 0.00815     | 14.35                     |
| DenseNet  | 6.92        | 0.01947     | 17.54                     |
| MobileNet | 2.71        | 0.00268     | 8.08                      |

### 結論

-------

在Jetson Nx平台中使用TensorRT可以減少演算法評估預測圖片的時間，但目前TensorRT僅支援一些常見的演算法，如自行開發的演算法並不一定能完整的使用TensorRT來進行預測加速。