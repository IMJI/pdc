# Параллельные и распределённые вычисления

**Улеев Данил ИВТ-13МО**

Для матрицы размером 1024х1024 лучшими результатами времени работы программ стали:

- **OpenMP**: 9.002 сек.
- **MPI**: 8.867 сек.
- **CUDA**: 0.075 сек.

Также, программа использующая CUDA была проверена на матрице размером 10000x10000. Время работы программы составило 153.311 сек.

## Графики зависимости времени работы программы от количества потоков

![OpenMP](./images/OpenMP.png?raw=true)

![MPI](./images/MPI.png?raw=true)

![Сравнение OpenMP и MPI](./images/MPI_OpenMP_comp.png?raw=true)
