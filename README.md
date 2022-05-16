# System of voice recognition

Реализован экспериментальный софт, на основе которого решается задача распознавания речи в реальном времени, что явялется основной целью курсовой работы "Создание систем голосового управления механического манипулятора".

## Аннотация курсовой работы:
В курсовой работе разработана и реализована система голосового управления без обработки звукового сигнала на сервере. Основное назначение системы – управление механическим манипулятором с помощью набора голосовых команд.
В работе приведены анализ и модификации основных алгоритмов нахождения участков сигнала, содержащих речь – нахождение голосовой активности в сигнале, что является одной из главных подзадач распознавания речи. Для распознавания команд была использована свободно распространяющаяся модель от Сбера «Голос». Ее описание и примеры использования также описаны в тексте курсовой работы.
Система голосового управления реализована по модульному принципу, что позволяет использовать её как основу для дальнейшего развития изученного подхода и построения систем распознавания речи, адаптиро- ванных под определенную задачу.

## Описание:
1) На ветке release в директория Voice_recognition_system находится Voice_recognition.py, с помощью чего решается задача распознавания "на лету" со звукозаписью, имитируя работу с микрофоном
2) Find_words.py -- нахождение звуковой активности алгоритмом zero crossing
3) Recognition_words.py -- распознавание найденных слов моделью от Сбера
4) Пример работы в папке tests
5) На ветке main можно найти другие примеры кода, где реализованы другие алгоритмы VAD и построение граификов
