# LEADERBOARD

Ниже победители по задачам курса GPGPU 2025.

Замеры делались на `Tesla V100-SXM2-16GB`: `15.67 TFLOPS FP32`, `897 GBytes/s` peak memory bandwidth.

## Task 03. Matrix Multiplication

| Place | Year | Participant | Team | Result | API | Link |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026 | Мовчанюк Павел | Военмех | 78121 GFlops | CUDA, WMMA | [PR #262](https://github.com/GPGPUCourse/GPGPUTasks2026/pull/262#issuecomment-6132196924) |
| 2 | 2026 | Пётр Маяцкий | ИТМО / AAM Systems | 71185 GFlops | CUDA, WMMA | [PR #358](https://github.com/GPGPUCourse/GPGPUTasks2026/pull/358#issuecomment-6129166683) |
| 3 | 2026 | Андрей Кашенин | СПбГУ / Яндекс | 64283 GFlops | CUDA, WMMA | [PR #405](https://github.com/GPGPUCourse/GPGPUTasks2026/pull/405#issuecomment-6127620133) |
| 4 | 2025 | Роберт Смайт | СПбГУ | 11720 GFlops | CUDA, WMMA | [PR #357](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/357/changes/6a5198e2aa136bb064fc99988d0717253ca28e4a) |
| 5 | 2025 | Mikhail Stulov | МФТИ / ТБанк | 9768 GFlops | OpenCL | [PR #399](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/399/changes/09fff64fd38e8bc3c63d894e014883e10e907b9d) |
| 6 | 2025 | Хулиган Серега | ИТМО | 7234 GFlops | OpenCL | [PR #309](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/309/changes/c1f4c050237a6637096f275555d594dbf30a1b97) |

P.S. `Tesla V100-SXM2-16GB`: `125 TFLOPS` Tensor Core FP16.

## Task 04. Prefix Sum

| Place | Year | Participant | Team | Result | API | Link |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026 | Павел Мовчанюк | Военмех | 748.98 GB/s | CUDA | [PR #482](https://github.com/GPGPUCourse/GPGPUTasks2026/pull/482#issuecomment-6276992417) |
| 2 | 2026 | Никита Богатырев | ИТМО | 745.12 GB/s | CUDA | [PR #511](https://github.com/GPGPUCourse/GPGPUTasks2026/pull/511#issuecomment-6274125945) |
| 3 | 2026 | Андрей Кашенин | СПбГУ / Яндекс | 744.05 GB/s | CUDA | [PR #496](https://github.com/GPGPUCourse/GPGPUTasks2026/pull/496#issuecomment-6262480925) |
| 4 | 2025 | Роберт Смайт | СПбГУ | 350.40 GB/s | CUDA | [PR #478](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/478/changes/0cfaca0c3973127a9b618e19202152ce43f63b7f) |
| 5 | 2025 | Илья Коннов | ИТМО / Яндекс | 295.33 GB/s | OpenCL | [PR #499](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/499/changes/70eb953c6788a84fdf971d20cef4e004c8909491) |
| 6 | 2025 | Mikhail Stulov | МФТИ / ТБанк | 278.85 GB/s | OpenCL | [PR #474](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/474/changes/0e41641564b8c6a78deb381c5cf7de3854534b86) |

## Task 05. Radix Sort

| Place | Year | Participant | Team | Result | API | Link |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026 | Павел Мовчанюк | Военмех | 16685.6 uint millions/s | CUDA | [PR #495](https://github.com/GPGPUCourse/GPGPUTasks2026/pull/495#issuecomment-6391185696) |
| 2 | 2026 | cub::DeviceRadixSort | NVIDIA | 15997.1 uint millions/s | CUDA | [PR #550](https://github.com/GPGPUCourse/GPGPUTasks2026/pull/550#issuecomment-6369934890) |
| 3 | 2026 | Никита Богатырев | ИТМО | 11105.3 uint millions/s | CUDA | [PR #609](https://github.com/GPGPUCourse/GPGPUTasks2026/pull/609#issuecomment-6426440143) |
| 4 | 2026 | Санджар Сатлыков | ВШЭ / YADRO | 10436.1 uint millions/s | CUDA | [PR #593](https://github.com/GPGPUCourse/GPGPUTasks2026/pull/593#issuecomment-6426414597) |
| 5 | 2025 | Роберт Смайт | СПбГУ | 3150.26 uint millions/s | CUDA | [PR #545](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/545/changes/cb7102356ba2f72dea488695401e3d7edb7f8872) |
| 6 | 2025 | Andrew Ratkov | СПбГУ | 1525.03 uint millions/s | OpenCL | [PR #568](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/568/changes/8fc00235a51dc14ee75f1c379266cc8743797027) |
| 7 | 2025 | Pribytkov Fedor | СПбГУ | 370.38 uint millions/s | OpenCL | [PR #569](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/569/changes/e6f2dd9969009f232c7a7aa1b21d477b0b7ba440) |

## Task 06. Merge Sort

| Place | Year | Participant | Team | Result | API | Link |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026 | Антон Тасаков | ИТМО | 4385.88 uint millions/s | OpenCL | [PR #658](https://github.com/GPGPUCourse/GPGPUTasks2026/pull/658#issuecomment-6560908843) |
| 2 | 2026 | Егор Коновалов | ИТМО | 3308.06 uint millions/s | OpenCL | [PR #669](https://github.com/GPGPUCourse/GPGPUTasks2026/pull/669#issuecomment-6579983302) |
| 3 | 2026 | Павел Мовчанюк | Военмех | 3288.23 uint millions/s | CUDA | [PR #688](https://github.com/GPGPUCourse/GPGPUTasks2026/pull/688#issuecomment-6587824138) |
| 4 | 2025 | Артём Батыгин | ВШЭ | 2074.01 uint millions/s | OpenCL | [PR #639](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/639/changes/e8af772fc73eb596841c7a39e3076d794768d9b2) |
| 5 | 2025 | Mikhail Stulov | МФТИ / ТБанк | 2054.23 uint millions/s | OpenCL | [PR #645](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/645/changes/2013a028ee0f3baa4fc5838089fde4897be92bf5) |
| 6 | 2025 | Роберт Смайт | СПбГУ | 1216.70 uint millions/s | CUDA | [PR #625](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/625/changes/393c73bb8e0a3824870f9e48af85c43f146a25fa) |

## Task 07. SpMV

| Place | Year | Participant | Team | Result (average over 5 cases) | API | Link |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2025 | Вячеслав Григорович | ИТМО | 47559.94 uint millions/s | OpenCL | [PR #680](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/680/changes/23a70c6c68dde9672ea115a995d38ab60eb6a1d9) |
| 2 | 2025 | Тяньшэн Цю | ИТМО | 34705.36 uint millions/s | OpenCL | [PR #647](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/647/changes/4ea307e7344961c857ab9a3cd3791dbca08b0ec5) |
| 3 | 2025 | Артём Батыгин | ВШЭ | 33163.08 uint millions/s | OpenCL | [PR #666](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/666/changes/16b8816077ad12bef47ff10cade6e9bbc52bde40) |

## Task 08. Ray Tracing

| Place | Year | Participant | Team | Result | API | Link |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2025 | Тяньшэн Цю | ИТМО | 215844 coolness = 279.48 MTris/s x 772.32 MRays/s | OpenCL | [PR #812](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/812/changes/a1429be138ae54c5d8fd6b4da30b330acf7613da) |
| 2 | 2025 | Ostapenko Vladislav | ВШЭ | 201689 coolness = 316.55 MTris/s x 637.16 MRays/s | OpenCL | [PR #803](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/803/changes/b9a15e181bf39927b585f40e061bab0a9ac1f3e6) |
| 3 | 2025 | Sanan Kornyakov | ВШЭ | 171595 coolness = 216.56 MTris/s x 792.38 MRays/s | OpenCL | [PR #813](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/813/changes/fce74a1b6ee1bf1a1bffe2aa8d846c95ccd922af) |
