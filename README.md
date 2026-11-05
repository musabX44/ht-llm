# ht-llm

HT-LLM, C ile yazılmış, CPU odaklı ve seyrek etkinleşen hiyerarşik ağaç dil modeli prototipidir. Eğitim ve demo/benchmark modlarını içerir.

## Derleme

GCC ve OpenMP runtime gerekir. Derleme sırasında GCC'nin bağımsız olduğunu doğrulayabildiği döngüler çok çekirdekli çalıştırılır; ardışık SGD adımları model ağırlıklarını paylaştığından sıralı kalır.

```sh
make clean all THREADS=4
./ht_llm --train dataset.txt --epochs 5
```

`THREADS` derleme zamanında hedeflenen iş parçacığı sayısıdır. Örneğin `make clean all THREADS=8`. Değeri değiştirdikten sonra `make clean` kullanarak tüm dosyaların yeni ayarlarla derlendiğinden emin olun.

GCC'nin güvenli paralelleştirme analizi muhafazakârdır; bağımlı eğitim adımları otomatik olarak paralelleştirilmez. Bu nedenle gerçek hızlanma döngülere ve veri kümesine bağlıdır. AVX2/FMA etkinleştirmesi mevcut Makefile ayarı olarak korunmuştur.
