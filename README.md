## MRS battey competition 紀錄

* kaggle連結[按此](https://www.kaggle.com/competitions/beep-competition/overview)
* 本模型使用的pytorch version: 1.11

### 資料說明

資料來源為Severson磷酸鋰鐵資料集的subset。從127顆電池中，每顆電池各挑選50連續循環作為輸入，預測目標則是該循環對應的電池剩餘使用壽命(Remain Useful Life, RUL)。輸入資料格式如下：

| **null**  | **voltage** | **test_time** | **discharge_capacity** | **discharge_energy** | **internal_resistance** | **cycle_index** | **current** |
|-----------|-------------|---------------|------------------------|----------------------|-------------------------|-----------------|-------------|
| **0**     | 3.583984    | 0.000000      | 0.000007               | 0.000025             | 0.015667                | 0               | -0.640613   |
| **1**     | 3.585586    | 0.000000      | 0.000007               | 0.000025             | 0.015667                | 0               | NaN         |
| **2**     | 3.587187    | 0.000000      | 0.000007               | 0.000025             | 0.015667                | 0               | NaN         |
| **3**     | 3.588789    | 0.000000      | 0.000007               | 0.000025             | 0.015667                | 0               | NaN         |
| **4**     | 3.590390    | 0.000000      | 0.000007               | 0.000025             | 0.015667                | 0               | NaN         |
| **...**   | ...         | ...           | ...                    | ...                  | ...                     | ...             | ...         |
| **49995** | 2.006406    | 797.709562    | 0.974379               | 2.955484             | 0.015613                | 49              | -4.400597   |
| **49996** | 2.004805    | 797.757812    | 0.974601               | 2.955938             | 0.015613                | 49              | -4.400666   |
| **49997** | 2.003203    | 809.350494    | 0.979560               | 2.965858             | 0.015613                | 49              | -3.268320   |
| **49998** | 2.001602    | 821.625366    | 0.984799               | 2.976336             | 0.015613                | 49              | -2.069057   |
| **49999** | 2.000000    | 898.679041    | 0.998511               | 3.003761             | 0.015613                | 49              | -0.277541   |

其中，每循環有1000筆數據跟7種特徵，但本模型使用的特徵只有discharge QV curve(voltage和discharge_capacity)。最終的目標是用108筆數據的訓練資料集訓練出在19筆數據的測試資料集上達到最低誤差的模型。  

### code說明

* [preprocessing.py](preprocessing.py): 用於將主辦方提供之json檔轉換為numpy array的形式，並簡化QV curve數據及編寫pytorch dataset
* [discharge_model.py](discharge_model.py): 以pytorch編寫模型架構
* [qv_curve_cnn_main.py](qv_curve_cnn_main.py): 執行模型訓練
* [utils.py](utils.py): 訓練及測試模型使用的工具
* [testing_result.py](testing_result.py): 預測結果視覺化及預測結果csv檔輸出
* [training_example.ipynb](training_example.ipynb): 完整的訓練流程及視覺化範本
* [model_evaluation_example.ipynb](model_evaluation_example.ipynb): 測試流程及視覺化範本


