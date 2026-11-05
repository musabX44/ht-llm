# ht-llm

HT-LLM, C ile yazılmış, CPU odaklı ve seyrek etkinleşen hiyerarşik ağaç dil modeli prototipidir. Eğitim ve demo/benchmark modlarını içerir.

## Çok çekirdekli CPU eğitimi

GCC ve OpenMP runtime gerekir. Derleyici, eğitimdeki bağımsız sayısal döngüleri (özellikle output-head ağırlık/gradyan güncellemelerini) OpenMP ile çok çekirdeğe dağıtabilir. Birbirine bağlı next-token SGD adımları ise ağırlık güncellemelerinin sırasını korumak için seri kalır; paralellik her adımın içindeki güvenli hesaplamalardadır.

```sh
make clean
make THREADS=8
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE ./ht_llm --train dataset.txt --epochs 5
```

`THREADS` derleyicinin oluşturabileceği paralel döngü iş parçacığı üst sınırını, `OMP_NUM_THREADS` ise çalışma anındaki iş parçacığı sayısını belirler. Bu iki değeri uyumlu ayarlayın. Örneğin 4 çekirdek için `make clean && make THREADS=4`, ardından `OMP_NUM_THREADS=4 ./ht_llm --train dataset.txt`.

GCC'nin otomatik paralelleştirme analizi bağımlı döngüleri paralelleştirmez. Bu tasarım, SGD adımlarını eşzamanlı çalıştırıp model ağırlıklarında race condition oluşturmaktan kaçınır. Gerçek hızlanma GCC sürümüne, işlemciye ve veri kümesine bağlıdır; `--verbose` kapalıyken ölçüm yapın.

## Derleme ve çalıştırma

```sh
make clean all
./ht_llm --train dataset.txt --epochs 5
```
