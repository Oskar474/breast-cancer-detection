
projekt dotyczy predykcji wystąpienia raka piersi na podstawie danych tabularycznych.\
dataset - Breast Cancer Dataset - https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

# Analiza danych

Dane podzieliłem na zbiór treningowy oraz testowy w proporcjach 7:3 

## Informacyjne

id – unikalny identyfikator próbki\
diagnosis – diagnoza (M = nowotwór złośliwy, B = łagodny)

## Średnie wartości (mean)

radius_mean – średni promień guza\
texture_mean – średnia tekstura (nieregularność powierzchni)\
perimeter_mean – średni obwód guza\
area_mean – średnia powierzchnia guza\
smoothness_mean – gładkość krawędzi (im mniejsze wahania, tym większa gładkość)\
compactness_mean – zwartość guza (obwód² / powierzchnia)\
concavity_mean – wklęsłość konturu guza\
concave points_mean – liczba wklęsłych punktów konturu\
symmetry_mean – symetria kształtu\
fractal_dimension_mean – złożoność krawędzi guza

## Odchylenia standardowe (se – standard error)

radius_se – zmienność promienia\
texture_se – zmienność tekstury\
perimeter_se – zmienność obwodu\
area_se – zmienność powierzchni\
smoothness_se – zmienność gładkości\
compactness_se – zmienność zwartości\
concavity_se – zmienność wklęsłości\
concave points_se – zmienność punktów wklęsłych\
symmetry_se – zmienność symetrii\
fractal_dimension_se – zmienność złożoności krawędzi

## Najgorsze wartości (worst – największe obserwowane)

radius_worst – największy promień guza\
texture_worst – największa nieregularność tekstury\
perimeter_worst – największy obwód\
area_worst – największa powierzchnia\
smoothness_worst – najmniejsza gładkość (największe wahania)\
compactness_worst – największa zwartość\
concavity_worst – największa wklęsłość\
concave points_worst – najwięcej punktów wklęsłych\
symmetry_worst – największa asymetria\
fractal_dimension_worst – najbardziej nieregularna krawędź

# Opis stosowanych modeli

Random Forest - Zespół wielu drzew decyzyjnych, z których każde uczy się na losowym podzbiorze danych i cech. Ostateczna decyzja to głosowanie wszystkich drzew.
XGBoost - Zaawansowany algorytm boostingowy budujący drzewa sekwencyjnie, gdzie każde kolejne poprawia błędy poprzedniego.
Logistic Regression - Model liniowy obliczający prawdopodobieństwo przynależności do klasy za pomocą funkcji logistycznej.
Decision Tree - Model podejmujący decyzje poprzez serię prostych reguł (podziałów danych na podstawie cech).

Simple_MLP:
```python 
   model = Sequential([
        Dense(32, activation='relu', input_shape=(30,)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name = "simple_MLP")
```
Learning rate: 0.0005, Optimizer: Adam, loss: binary crossentropy\
![simple_val.png](plots%2Fsimple_val.png)![simple_loss.png](plots%2Fsimple_loss.png)

Tuned_NN:
```python 
 model = Sequential([
        Dense(40, activation='relu', input_shape=(30,)),
        Dense(40, activation='relu'),
        Dropout(rate=0.4),
        Dense(40, activation='relu'),
        Dropout(rate=0.3),
        Dense(1, activation='sigmoid')
    ], name="tuned_NN")
```
Learning rate: 0.001, Optimizer: Adam, loss: binary crossentropy\
![tuned_acc.png](plots%2Ftuned_acc.png)![tuned_loss.png](plots%2Ftuned_loss.png)

Paper_NN:
```python 
    model = Sequential([
        Dense(16, activation='relu', input_shape=(30,)),
        Dropout(0.4),
        Dense(16, activation='relu'),
        Dropout(0.4),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name="paper_NN")
```
Learning rate: 0.001, Optimizer: Adam, loss: binary crossentropy\
![paper_acc.png](plots%2Fpaper_acc.png)![paper_loss.png](plots%2Fpaper_loss.png)

CNN
```python 
    model = Sequential([
        Reshape((25, 1), input_shape=(25,)),

        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),

        Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),

        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),

        Dense(1, activation='sigmoid')
    ], name="cnn")
```
Learning rate: 0.001, Optimizer: Adam, loss: binary crossentropy\
![cnn_acc.png](plots%2Fcnn_acc.png)![cnn_loss.png](plots%2Fcnn_loss.png)

CNN_small
```python 
model = Sequential([
        Reshape((25, 1), input_shape=(25,)),

        Conv1D(filters=8, kernel_size=2, activation='relu', padding='same'),
        BatchNormalization(),

        Flatten(),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(8, activation='relu'),

        Dense(1, activation='sigmoid')
    ], name="cnn_small")

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
```
Learning rate: 0.001, Optimizer: Adam, loss: binary crossentropy

# Proces powstawania projektu:
Metryki jakich użyłem do oceny modeli to:
- accuracy
- f1 score
- precision
- recall
- balanced accuracy

Z powodu dostrzegalnych wachań pomiędzy poszczególnymi próbami każdy poniższy wynik jest uśrednieniem wartości tych metryk z 10 prób\
W dalszych wnioskach metryki jakie uznałem za najważniejsze co za tym idzie kierowałem się nimi w ocenie modeli to recall oraz balanced accuract\
recall jest bardzo istotną metryką w wykrywaniu schorzeń ponieważ zapewnia że przypadki prawdziwie pozytywne nie są błędnie klasyfikowane\
balanced accuracy jest dobrą metryką ponieważ dataset nie jest idealnie zbalansowany

## Najpierw bez żadnej prepraracji danych postanowiłem przetestować powyższe algorytmy oraz Prostą sieć neuronową jako baseline

![raw_data.png](images%2Fraw_data.png)

Tutaj najlepiej wypadają XGB oraz Random Forest, słaby wynik sieci neuronowej jest jak najbardziej oczekiwany bez skalowania danych
## Następnie, jako że wszystkie kolumny były wartościami ciągłymi użyłem skalowania za pomocą StandardScaler udostępnionego przez scikit-learn

![scaled_data.png](images%2Fscaled_data.png)

Skalowanie danych przyniosło znaczącą poprawę w przypadku sieci neuronowej oraz Regresji Logistycznej

## Teraz postanowiłem sprawdzić dwie kolejne architektury jedną z istniejącego już rozwiązania, opisaną wcześniej jako paper_NN na kaggle w celach porównawczych oraz jedną uzyskaną poprzez keras Tuner

Parametry tunera:
- warstwa wejściowa 8-64 neurony na wyjściu, skok 16
- od 1 do 4 warstw ukrytych o parametrach 8-64 neurony, skok 16
- dropout dodawany po każdej warstwie ukrytej o parametrach 0.1-0.5, skok 0.1
- learning rate z możliwymi ustawieniami: 1e-2, 1e-3, 1e-4
- tuner uruchomiłem z parametrami: 100 architektur do wypróbowania, 3 próby na architekture, 50 epok

Najlepsza architektura jaką znalazł tuner, wcześniej opisana jako tuned_NN:
```python 
 model = Sequential([
        Dense(40, activation='relu', input_shape=(30,)),
        Dense(40, activation='relu'),
        Dropout(rate=0.4),
        Dense(40, activation='relu'),
        Dropout(rate=0.3),
        Dense(1, activation='sigmoid')
    ], name="tuned_NN")
```
Learning rate: 0.001

![scaled_data_tuned_paper.png](images%2Fscaled_data_tuned_paper.png)

Rezultaty z działania tunera okazały się znaczące, architektura otrzymana w ten sposób wypadła najlepiej pośród innych sieci neuronowych\
model paper_NN jest nieznacznie gorszy niż ten uzyskany przeze mnie natomiast pierwotny prosty model zostaje daleko w tyl,\
nadal jednak regresja logistyczna radzi sobie najlepiej w balanced accuracy, choć ma gorszy recall 

## Następnie postanowiłem zastosować feature extraction, odcinając część kolumn z datasetu które miały znikomą bądź negatywną korelację z diagnozą. Wartości korelacji obliczone zostały za pomocą współczynnika korelacji Pearsona używajać funkcji Dataframe.corr()

![feature_extraction.png](images%2Ffeature_extraction.png)

zastosowana metoda ekstrakcji cech zaowocowała poprawieniem wyników każdej z architektur sieci neuronowych oraz dużą poprawą wyniku regresji logistycznej \
ciekawą obserwacją jest to że najprostszy model simple_MLP prześcignął paper_NN oraz znacząco zbliżył się do tuned_NN, takie zjawisko ma jednak sens,\
zmniejszyła się liczba cech w datasecie co sprawiło, że archtiektura najprostszego modelu stała się wystarczająca dla danych jakie otrzymuje.

## Na koniec w celu wypróbowania innego typu arhchitektur stworzyłem dwa modele jednowymiarowych sieci konwolucyjnych

![cnns.png](images%2Fcnns.png)

Mniejszy model, który stworzyłem jako drugi okazał się lepszy, pierwszy był zbyt rozbudowany dla prostego datasetu którego używam,\
Wyniki były bardzo zbliżone do dotychczas najlepszego tuned_NN i lepsze od innych architektur 

# Wnioski końcowe

- w przypadku prostych zbiorów danych tabularycznych należy ostrożnie zwiększaść głębokość modeli oraz wielkość warstw, bardzo łatwo można stworzyć architekturę zbyt złożoną
- ekstrakcja cech jest dobrą techniką zarówno w przypadku sieci neuronowych, jak i w przypadku regresji logistycznej
- projekt uważam za udany, wyniki modeli były satysfakcjonujące choć to regresja logistyczna okazała się najlepsza