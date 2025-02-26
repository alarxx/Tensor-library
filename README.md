# Tensor-library (C++)

## Tensor 

Tensor - array of Tensors, except rank-0 Tensor (scalar)
```
[Tensor, Tensor, Tensor, Tensor]  
   |       |       |       |  
   |       |     [...]   [...]  
   |  [Tensor, Tensor, ...]  
[Tensor, Tensor...]  
```

---

## Features

- Tensors


## Licence

Tensor-library is licensed under the terms of [MPL-2.0](https://mozilla.org/MPL/2.0/), which is simple and straightforward to use, allowing this project to be combined and distributed with any proprietary software, even with static linking. If you modify, only the originally covered files must remain under the same MPL-2.0.

License notice:
```
SPDX-License-Identifier: MPL-2.0
--------------------------------
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file,
You can obtain one at https://mozilla.org/MPL/2.0/.

This file is part of the Tensor-library:
https://github.com/alarxx/Tensor-library

Description: <Description>

Provided “as is”, without warranty of any kind.

Copyright © 2025 Alar Akilbekov. All rights reserved.

Third party copyrights are property of their respective owners.
```

---

Если вы собираетесь модифицировать MPL покрытые файлы (IANAL):
- Ваш fork проект = MPL Covered files + Not Covered files (1.7).
- MPL-2.0 is file-based weak copyleft действующий только на Covered файлы, то есть добавленные вами файлы и исполняемые файлы, полученные из объединенения с вашими, могут быть под любой лицензией (3.3).
- (но под copyleft могут подпадать и новые файлы в которых copy-paste-нули код из Covered) (1.7).
- Покрытыми лицензией (Covered) считаются файлы с license notice (e.g. .cpp, .hpp) и любые исполняемые виды этих файлов (e.g. .exe, .a, .so) (1.4).
- You may not remove license notices (3.4), как и в MIT, Apache, BSD (кроме 0BSD) etc.
- При распространении любой Covered файл должен быть доступен, но разрешено личное использование или только внутри организации (3.2).
- Если указан Exhibit B, то производную запрещается лицензировать с GPL.
- Contributor дает лицензию на любое использование конкретной интеллектуальной собственности (patent), которую он реализует в проекте (но не trademarks).

Эти разъяснения условий не меняют и не вносят новые юридические требования к MPL.

## Contact

Alar Akilbekov - alar.akilbekov@gmail.com

## References:
- Weidman, S. (2019). Deep learning from scratch: Building with Python from first principles (First edition). O’Reilly Media, Inc.
- Patterson, J., & Gibson, A. (2017). Deep learning: A practitioner’s approach (First edition). O’Reilly.
- Евгений Разинков. (2021). Лекции по Deep Learning. https://www.youtube.com/playlist?list=PL6-BrcpR2C5QrLMaIOstSxZp4RfhveDSP
- Raschka, S., Liu, Y., Mirjalili, V., & Dzhulgakov, D. (2022). Machine learning with PyTorch and Scikit-Learn: Develop machine learning and deep learning models with Python. Packt.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. The MIT press.
- Rashid, T. (2016). Make Your own neural network. CreateSpace Independent Publishing Platform.
